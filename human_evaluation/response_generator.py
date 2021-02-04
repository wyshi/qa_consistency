from torch import cuda
from parlai.core.agents import create_agent_from_model_file
import parlai.tasks.q_function_consistency.utils as utils
from parlai.core.teachers import create_task_agent_from_taskname
from copy import deepcopy

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import json
from parlai.tasks.q_function_consistency.qa_model import InteractiveQageneratorTeacher
from parlai.utils import logging
from parlai.tasks.q_function_consistency.utils import RAW_BLENDER_PATH, MAX_CANDIDATE
from parlai.tasks.q_function_consistency.human_evaluation.safety_agent import SafetyAgent, NO_MATCH, STRING_MATCHED, CLASSIFIER_MATCHED


with open(f"{RAW_BLENDER_PATH}.opt", "r") as fh:
    blender_model_opt = json.load(fh)
    DELIMITER = blender_model_opt['delimiter']

class Raw_Blender():
    def __init__(self, has_classifier, model_checkpoint, classifier_checkpoint, qa_model_gpu_device, qa_generator=None, safety_classifier=None):
        self.model_checkpoint = model_checkpoint
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu_num = 0

        self.has_classifier = has_classifier
        self.classifier_checkpoint = classifier_checkpoint
        self.classifier = None
        self.classifier_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.classifier_gpu_num = 0



        self.qa_model_gpu_device = qa_model_gpu_device
        if qa_generator is None:
            self.qa_generator = InteractiveQageneratorTeacher('/home/wyshi/ParlAI/parlai/tasks/q_function_consistency/models/t52020_11_12_18_43noanswer_more_best_1.694010293717478.pth', 
                                qa_model_gpu_device=self.qa_model_gpu_device)
        else:
            self.qa_generator = qa_generator

        if safety_classifier is None:
            self.safety_classifier = self.load_safety_classifier()
        else:
            self.safety_classifier = safety_classifier

    def load_safety_classifier(self):
        return SafetyAgent({'safety': 'all'})

    def load_model(self, gpu_num):
        if self.model is None:
            opt_overrides = {}
            self.gpu_num = gpu_num
            opt_overrides['gpu'] = gpu_num
            opt_overrides['datatype'] = 'test'
            opt_overrides['inference'] = 'nucleus'
            opt_overrides['skip_generation'] = False
            
            self.model = create_agent_from_model_file(self.model_checkpoint, opt_overrides=opt_overrides)
            logging.info("load Raw Blender model from:{}".format(self.model_checkpoint))
            logging.info("allocate Raw Blender model to gpu_{}".format(gpu_num))

    def load_classifier(self, gpu_num):
        if self.has_classifier:
            if self.classifier is None:
                opt_overrides = {}
                self.classifier_gpu_num = gpu_num
                opt_overrides['gpu'] = gpu_num
                opt_overrides['datatype'] = 'test'
                # opt_overrides['inference'] = 'nucleus'
                opt_overrides['skip_generation'] = False
                
                self.classifier = create_agent_from_model_file(self.classifier_checkpoint, opt_overrides=opt_overrides)
                teacher_for_classifier_opt = deepcopy(self.classifier.opt)
                teacher_for_classifier_opt.update({"build_data_or_not": False})
                self.teacher_for_classifier = create_task_agent_from_taskname(teacher_for_classifier_opt)[0]
                logging.info("load classifier from:{}".format(self.classifier_checkpoint))
                logging.info("allocate classifier to gpu_{}".format(gpu_num))
        else:
            self.classifier = None

    def set_model(self, model):
        self.model = model

    def set_classifier(self, classifier):
        self.classifier = classifier
        teacher_for_classifier_opt = deepcopy(self.classifier.opt)
        teacher_for_classifier_opt.update({"build_data_or_not": False})
        self.teacher_for_classifier = create_task_agent_from_taskname(teacher_for_classifier_opt)[0]

    def _build_up_model_input(self, history, user_text):
        text = DELIMITER.join(history + [user_text])
        # text = text.lower()
        return text

    def build_context_for_classifier(self, context_list):
        # logging.info(f"context list: {context_list}")
        try:
            assert len(context_list)%2 == 1
        except:
            import pdb
            pdb.set_trace()
        context = ""
        for i, turn in enumerate(context_list):
            if i % 2 == 0:
                turn = f"human: {turn}\n"
            else:
                turn = f"bot: {turn}\n"
            context += turn

        context = context.strip("\n")
        # for entry_idx in range(0, len(context_list), 2):
        #     human_turn = f"human: {context_list[entry_idx]}\n"
        #     agent_turn = f"bot: {context_list[entry_idx+1]}\n"
        #     context += human_turn  # + agent_turn
        #     # action = self._get_base_msg(full_eps, entry_idx, context, agent_turn)
        #     context += agent_turn
        # # finish the tail
        # human_turn = f"human: {context_list[entry_idx]}\n"
        # context += human_turn

        return context

    def _decide_status(self, context, candidate):
        self.classifier.reset()
        persona = DELIMITER.join(context[:1])
        context = self.build_context_for_classifier(context[1:])
        full_eps = {'context': context,
                    'persona': persona,
                    'candidate': candidate,
                    'generated_human_response': None,
                    'generated_questions': []}
        full_eps = self.qa_generator.construct_message(full_eps)
        full_eps = self.teacher_for_classifier._construct_message(full_eps)
        logging.info(f"input to the clf: {full_eps['text']}")
        self.classifier.observe({'text': full_eps['text'], 'episode_done': True})
        is_good = self.classifier.act()['text'] == "good"
        assert type(is_good) is bool
        return is_good

        # act = {'text': text, 'episode_done': True}
        # self.model.observe(act)
        # response = self.model.act()['text']
        # pred_class, prob = [x.split(': ')[-1] for x in response.split('\n')]
        # pred_not_ok = self.classes[pred_class]  # check whether classified as NOT OK
        # prob = float(prob)  # cast string to float

        # return pred_not_ok, prob


    def process(self, history, user_text):
        # if not user_text:
        #     user_text = " [SEP] "
        torch.cuda.set_device(self.gpu_num)
        has_good_response = False
        good_cnt = 0
        bad_cnt = 0

        # if user text is safe
        user_offensive = self.safety_classifier.observe_and_act(user_text)
        if user_offensive in [STRING_MATCHED]:
            logging.warn(f'user offensive, {user_text}')
            logging.warn(utils.REPLY_TO_HUMAN_OFFENSIVE_MSG)
            return utils.REPLY_TO_HUMAN_OFFENSIVE_MSG, good_cnt, bad_cnt

        while not has_good_response:
            bot_offensive = None
            while bot_offensive is None or bot_offensive is True:
                logging.warn(f"------------------ reseting model {self.model}-------------------")
                self.model.reset()
                inputs = self._build_up_model_input(history, user_text)
                # logging.info("input to the raw blender:\n{}".format(inputs))
                logging.warn(f"------------------ model observing {self.model}-------------------")
                self.model.observe({'text': inputs, 'episode_done': True})
                logging.warn(f"------------------ model acting {self.model}-------------------")
                output = self.model.act()
                logging.warn(f"------------------ model acting finished {self.model}-------------------")

                if output is not None:
                    response_candidate = output['text']                
                    if self.safety_classifier.observe_and_act(response_candidate) not in [STRING_MATCHED, CLASSIFIER_MATCHED]:
                        bot_offensive = False
                    else:
                        bot_offensive = True
                        logging.warn(f'bot offensive: {response_candidate}')

                else:
                    return "Raw Blender SYSTEM ERROR!", good_cnt, bad_cnt


            if output is not None:
                response_candidate = output['text']
                history = history + [user_text]
                if self.has_classifier:
                    is_good = self._decide_status(context=history, candidate=response_candidate)
                else:
                    is_good = True
                

                if is_good:
                    good_cnt += 1
                    logging.info(f"good response!")
                    logging.info(f"{response_candidate}")
                    logging.info(f"-------turn end-------")
                    has_good_response = True
                    return output['text'], good_cnt, bad_cnt
                elif (good_cnt + bad_cnt) >= MAX_CANDIDATE:
                    logging.warn(f"bad response but reach max candidate!")
                    logging.warn(f"context: {history}")
                    logging.warn(f"response: {response_candidate}")
                    logging.info(f"-------turn end-------")
                    bad_cnt += 1
                    has_good_response = True
                    return output['text'], good_cnt, bad_cnt
                else:
                    logging.warn(f"bad response but not max yet!")
                    logging.warn(f"context: {history}")
                    logging.warn(f"response: {response_candidate}")
                    bad_cnt += 1
                    history = history[:-1]
                    logging.info(f"-------turn end-------")
                    has_good_response = False
            else:
                has_good_response = True
                return "Raw Blender SYSTEM ERROR!", good_cnt, bad_cnt

from flask import Flask, render_template, session, request, jsonify
import json
import os
import time
import random

from parlai.tasks.q_function_consistency.human_evaluation.raw_blender import RawBlenderBot
from parlai.tasks.q_function_consistency.utils import RAW_BLENDER_PATH
from parlai.tasks.blended_skill_talk.worlds import get_contexts_data
from parlai.utils import logging
from copy import deepcopy
from parlai.tasks.q_function_consistency.qa_model import InteractiveQageneratorTeacher
from parlai.core.agents import create_agent_from_model_file
from parlai.tasks.q_function_consistency.human_evaluation.safety_agent import SafetyAgent
from parlai.utils.safety import OffensiveStringMatcher, OffensiveLanguageClassifier
from parlai.agents.transformer.transformer import TransformerClassifierAgent

#### load models
def load_model(model_checkpoint, gpu_num):
    opt_overrides = {}
    opt_overrides['gpu'] = gpu_num
    opt_overrides['datatype'] = 'test'
    opt_overrides['inference'] = 'nucleus'
    opt_overrides['skip_generation'] = False    
    model = create_agent_from_model_file(model_checkpoint, opt_overrides=opt_overrides)
    logging.info("load Raw Blender model from:{}".format(model_checkpoint))
    logging.info("allocate Raw Blender model to gpu_{}".format(gpu_num))
    return model

def load_qa_model(qa_model_gpu_device):
    qa_generator = InteractiveQageneratorTeacher('/home/wyshi/ParlAI/parlai/tasks/q_function_consistency/models/t52020_11_12_18_43noanswer_more_best_1.694010293717478.pth', 
                                    qa_model_gpu_device=qa_model_gpu_device)
    return qa_generator

def load_safety_model(custom_model_file):
    from parlai.core.params import ParlaiParser

    parser = ParlaiParser(False, False)
    TransformerClassifierAgent.add_cmdline_args(parser, partial_opt=None)
    parser.set_params(
        model='transformer/classifier',
        model_file=custom_model_file,
        print_scores=True,
        data_parallel=False,
    )
    safety_opt = parser.parse_args([])
    return create_agent(safety_opt, requireModelExists=True)


# load models
# shared 
QA_GENERATOR_GPU = "cuda:0" # actual gpu 1
QA_GENERATOR = load_qa_model(qa_model_gpu_device=QA_GENERATOR_GPU)

# safety
SAFETY_CLASSIFIER = load_safety_model('/home/wyshi/ParlAI/data/models/dialogue_safety/single_turn/model')
SAFETY_AGENT = SafetyAgent({'safety': 'all'}) # actual gpu 1

# not shared
BLENDER_MODEL_GPU_1 = 2 # actual gpu 0
CLF_1_GPU = 2
BLENDER_MODEL_1 = load_model(model_checkpoint=RAW_BLENDER_PATH, gpu_num=BLENDER_MODEL_GPU_1)

BLENDER_MODEL_GPU_2 = 6
CLF_2_GPU = 6
BLENDER_MODEL_2 = load_model(model_checkpoint=RAW_BLENDER_PATH, gpu_num=BLENDER_MODEL_GPU_2)

BLENDER_MODEL_GPU_3 = 7
CLF_3_GPU = 7
BLENDER_MODEL_3 = load_model(model_checkpoint=RAW_BLENDER_PATH, gpu_num=BLENDER_MODEL_GPU_3)


SID_TO_PERSONA = {}
# Init the server
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
CLF_CKPT = "/home/wyshi/ParlAI/parlai/tasks/q_function_consistency/models/classifiers/8801/model"
chatbot = RawBlenderBot()
logging.info("loading raw blender model")
chatbot.load_response_generator(has_classifier=True, 
                                checkpoint=RAW_BLENDER_PATH, 
                                classifier_checkpoint=CLF_CKPT, 
                                gpu_num=BLENDER_MODEL_GPU_1, 
                                clf_gpu_num=CLF_1_GPU, 
                                qa_model_gpu_device=QA_GENERATOR_GPU,
                                qa_generator=QA_GENERATOR, 
                                blender_model=BLENDER_MODEL_1,
                                safety_classifier=SAFETY_AGENT)
MANAGER1 = {}

# 2nd model 
CLF_CKPT2 = "/home/wyshi/ParlAI/parlai/tasks/q_function_consistency/models/classifiers/b6d8/model"
chatbot2 = RawBlenderBot()
logging.info("loading raw blender model2")
chatbot2.load_response_generator(has_classifier=True, 
                                checkpoint=RAW_BLENDER_PATH, 
                                classifier_checkpoint=CLF_CKPT2, 
                                gpu_num=BLENDER_MODEL_GPU_2, 
                                clf_gpu_num=CLF_2_GPU, 
                                qa_model_gpu_device=QA_GENERATOR_GPU,
                                qa_generator=QA_GENERATOR, 
                                blender_model=BLENDER_MODEL_2, 
                                safety_classifier=SAFETY_AGENT)
MANAGER2 = {}

# 3rd model 
CLF_CKPT3 = ""
chatbot3 = RawBlenderBot()
logging.info("loading raw blender model3")
chatbot3.load_response_generator(has_classifier=False, 
                                checkpoint=RAW_BLENDER_PATH, 
                                classifier_checkpoint=CLF_CKPT3, 
                                gpu_num=BLENDER_MODEL_GPU_3, 
                                clf_gpu_num=CLF_3_GPU, 
                                qa_model_gpu_device=QA_GENERATOR_GPU,
                                qa_generator=QA_GENERATOR, 
                                blender_model=BLENDER_MODEL_3,
                                safety_classifier=SAFETY_AGENT)
MANAGER3 = {}



def build_persona_pool():
    with open("/home/wyshi/ParlAI/data/models/blender/blender_90M/model.opt", "r") as fh:
        persona_model_opt = json.load(fh)
    persona_model_opt.update({
        "datapath": '/home/wyshi/ParlAI/data',
        'include_personas': True,
    'safe_personas_only': True,
    'include_initial_utterances': False})
    persona_pool = get_contexts_data(opt=persona_model_opt, shared=None)
    return persona_pool

def initialize_one_persona():
    random.seed()
    p = random.choice(PERSONA_POOL)
    p = p[0]
    if len(p.split("\n")) == 3:
        p = "\n".join(p.split("\n")[:2])
    return [p]

PERSONA_POOL = build_persona_pool()


def get_chat_input(sid):
    if sid in MANAGER1.keys():
        history_with_entities = MANAGER1[sid]
    else:
        sid_persona = sid.split('+agent_name:')[0]
        if sid_persona in SID_TO_PERSONA:
            initial_persona = SID_TO_PERSONA[sid_persona]
        else:
            logging.warn(f"persona pool: {SID_TO_PERSONA}")
            initial_persona = initialize_one_persona()
            SID_TO_PERSONA[sid_persona] = deepcopy(initial_persona)
        history_with_entities = deepcopy(initial_persona) #+ ["Hi!"]
        MANAGER1[sid] = history_with_entities
    return history_with_entities

def set_chat_history(sid, history_with_entities, user_text, response, end_chat):
    if sid not in MANAGER1.keys():
        logging.info("Raw blender ERROR! sid is not in manager!")
    else:
        if history_with_entities:
            history_with_entities.append(user_text)
        history_with_entities.append(response)
        MANAGER1[sid] = history_with_entities
    if end_chat:
        logging.info(f"end of chat!\n{history_with_entities}")
        MANAGER1.pop(sid, None)

def get_chat_input_2(sid):
    if sid in MANAGER2.keys():
        history_with_entities = MANAGER2[sid]
    else:
        sid_persona = sid.split('+agent_name:')[0]
        if sid_persona in SID_TO_PERSONA:
            initial_persona = SID_TO_PERSONA[sid_persona]
        else:
            logging.warn(f"persona pool: {SID_TO_PERSONA}")
            initial_persona = initialize_one_persona()
            SID_TO_PERSONA[sid_persona] = deepcopy(initial_persona)
        history_with_entities = deepcopy(initial_persona) #+ ["Hi!"]
        MANAGER2[sid] = history_with_entities
    return history_with_entities

def set_chat_history_2(sid, history_with_entities, user_text, response, end_chat):
    if sid not in MANAGER2.keys():
        logging.info("Raw blender ERROR! sid is not in manager!")
    else:
        if history_with_entities:
            history_with_entities.append(user_text)
        history_with_entities.append(response)
        MANAGER2[sid] = history_with_entities
    if end_chat:
        logging.info(f"end of chat!\n{history_with_entities}")
        MANAGER2.pop(sid, None)

def get_chat_input_3(sid):
    if sid in MANAGER3.keys():
        history_with_entities = MANAGER3[sid]
    else:
        sid_persona = sid.split('+agent_name:')[0]
        if sid_persona in SID_TO_PERSONA:
            initial_persona = SID_TO_PERSONA[sid_persona]
        else:
            logging.warn(f"persona pool: {SID_TO_PERSONA}")
            initial_persona = initialize_one_persona()
            SID_TO_PERSONA[sid_persona] = deepcopy(initial_persona)
        history_with_entities = deepcopy(initial_persona) #+ ["Hi!"]
        MANAGER3[sid] = history_with_entities
    return history_with_entities

def set_chat_history_3(sid, history_with_entities, user_text, response, end_chat):
    if sid not in MANAGER3.keys():
        logging.info("Raw blender ERROR! sid is not in manager!")
    else:
        if history_with_entities:
            history_with_entities.append(user_text)
        history_with_entities.append(response)
        MANAGER3[sid] = history_with_entities
    if end_chat:
        logging.info(f"end of chat!\n{history_with_entities}")
        MANAGER3.pop(sid, None)

@app.route("/raw_blender", methods=['POST'])
def get_Response():
    """ Get response from the raw blender chatbot."""
    sid = request.json.get('sid')
    # context = request.json.get('context')
    user_text = request.json.get('user_text')
    if user_text == "":
        user_text = "Hi!"
        logging.info(f"-----= conversation starts! -------")
        logging.info(f"manager1: {MANAGER1}")
        logging.info(f"manager2: {MANAGER2}")
    logging.info(f"received user_text, {user_text}")
    logging.info(f"get message, sid={sid}")
    # logging.info(f"current persona: {SID_TO_PERSONA}")
    history_with_entities = get_chat_input(sid)
    response, end_chat, good_cnt, bad_cnt = chatbot.chat(history_with_entities, user_text)
    set_chat_history(sid, history_with_entities, user_text, response, end_chat)

    logging.info(f"good_cnt: {good_cnt}")
    logging.info(f"bad_cnt: {bad_cnt}")
    if end_chat:
        logging.info(f"end chat! {sid}, manager1")
        logging.info(MANAGER1)
        logging.info(f"=======================================\n")

    return jsonify({"response": response, "end_chat": end_chat, "good_cnt": good_cnt, "bad_cnt": bad_cnt})

@app.route("/raw_blender_baseline_classifier", methods=['POST'])
def get_Response_baseline_classifier():
    """ Get response from the raw blender chatbot."""
    sid = request.json.get('sid')
    # context = request.json.get('context')
    user_text = request.json.get('user_text')
    if user_text == "":
        user_text = "Hi!"
        logging.info(f"conversation starts!")
        logging.info(f"manager1: {MANAGER1}")
        logging.info(f"manager2: {MANAGER2}")
    logging.info(f"received user_text, {user_text}")
    logging.info(f"get message, sid={sid}")
    # logging.info(f"current persona: {SID_TO_PERSONA}")
    history_with_entities = get_chat_input_2(sid)
    # print(f"history in get_response: {history_with_entities}\n")
    response, end_chat, good_cnt, bad_cnt = chatbot2.chat(history_with_entities, user_text)
    set_chat_history_2(sid, history_with_entities, user_text, response, end_chat)
    # print(f"history in get_response after: {history_with_entities}\n")

    logging.info(f"good_cnt: {good_cnt}")
    logging.info(f"bad_cnt: {bad_cnt}")
    if end_chat:
        logging.info(f"end chat! {sid}, manager2")
        logging.info(MANAGER2)
        logging.info(f"=======================================\n")

    return jsonify({"response": response, "end_chat": end_chat, "good_cnt": good_cnt, "bad_cnt": bad_cnt})

@app.route("/raw_blender_baseline_no_classifier", methods=['POST'])
def get_Response_baseline_no_classifier():
    """ Get response from the raw blender chatbot."""
    sid = request.json.get('sid')
    # context = request.json.get('context')
    user_text = request.json.get('user_text')
    if user_text == "":
        user_text = "Hi!"
        logging.info(f"conversation starts!")
        logging.info(f"manager1: {MANAGER1}")
        logging.info(f"manager2: {MANAGER2}")
    logging.info(f"received user_text, {user_text}")
    logging.info(f"get message, sid={sid}")
    # logging.info(f"current persona: {SID_TO_PERSONA}")
    history_with_entities = get_chat_input_3(sid)
    # print(f"history in get_response: {history_with_entities}\n")
    response, end_chat, good_cnt, bad_cnt = chatbot3.chat(history_with_entities, user_text)
    set_chat_history_3(sid, history_with_entities, user_text, response, end_chat)
    # print(f"history in get_response after: {history_with_entities}\n")

    logging.info(f"good_cnt: {good_cnt}")
    logging.info(f"bad_cnt: {bad_cnt}")
    if end_chat:
        logging.info(f"end chat! {sid} manager 3")
        logging.info(MANAGER3)
        logging.info(f"=======================================\n")

    return jsonify({"response": response, "end_chat": end_chat, "good_cnt": good_cnt, "bad_cnt": bad_cnt})

if __name__ == '__main__':
    """ Run the app. """
    # socketio.run(app, port=3322)
    app.run(host='0.0.0.0', port=9876)

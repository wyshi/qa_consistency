from transformers import pipeline
import nltk
import time

from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration
import parlai.tasks.q_function_consistency.utils as utils

import argparse
import torch

# interactively get qa information
class InteractiveQageneratorTeacher:
    """
    special teacher to generate qa pairs, shouldn't be used for training

    parlai dd -t mmmmmm:q_function_consistency:bot_only_allclasses_interactive_qagenerator --verbose True
    """

    def __init__(self, qa_model_path, qa_model_gpu_device):
        self.qa_model_gpu_device = qa_model_gpu_device
        self.nlp = load_model_for_eval(qa_model_path, qa_model_gpu_device)

    def construct_message(self, full_eps):
        # full_eps = self._join_generated_info(full_eps)
        persona = full_eps['persona'].replace("your persona:", "bot:")
        context = full_eps['context']
        candidate = full_eps['candidate']
        future_human_response = full_eps['generated_human_response']
        generated_questions = full_eps['generated_questions']

        (
            qa_pair,
            answer_bot_persona,
            (answer_human_context, answer_bot_context),
            (answer_human_candidate, answer_bot_candidate),
            status,
        ) = answer_one_dialogue(
            get_answer=get_answer_unified_qa,
            nlp=self.nlp,
            persona=persona,
            context=context,
            candidate=candidate,
            future_human_response=future_human_response,
            verbose=False,
            generated_questions=generated_questions,
            device=self.qa_model_gpu_device,
        )

        # full_eps['text'] = self._generate_context(
        #     full_eps['persona'], full_eps['context'], full_eps['candidate']
        # )

        full_eps['answer_bot_persona'] = answer_bot_persona
        full_eps['answer_human_context'] = answer_human_context
        full_eps['answer_human_candidate'] = answer_human_candidate
        full_eps['answer_bot_context'] = answer_bot_context
        full_eps['answer_bot_candidate'] = answer_bot_candidate
        # full_eps['answer_human_next_response'] = list(
        #     qa_pair['human']['future_response'].values()
        # )
        full_eps['original_status'] = status
        full_eps['qa_pair'] = qa_pair

        return full_eps




def load_pretained_model_and_tokenizer(
    base_model: str, model_dict_path: str, gpu_device: str, eval=False, 
):
    '''
    Load pretainted T5 model on UnifiedQA
    base_model: base model name for T5
    model_dict_path: trained model checkpoint for unifiedQA
    '''
    tokenizer = T5Tokenizer.from_pretrained(base_model)
    model = T5ForConditionalGeneration(T5Config.from_pretrained(base_model))

    if eval:
        model = torch.load(model_dict_path, map_location=gpu_device)
    else:
        load_tf_weights_in_t5(model, None, model_dict_path)

    return tokenizer, model


def load_model_for_eval(model_path, gpu_device):
    if model_path == "~/ParlAI/data/models/t5/large/models_large_model.ckpt-1101200":
        is_TF_model = True
    else:
        is_TF_model = False
    tokenizer, model = load_pretained_model_and_tokenizer(
        't5-large', model_path, gpu_device=gpu_device, eval=(not is_TF_model), 
    )
    model.eval()
    model.to(gpu_device)
    nlp = (model, tokenizer)

    return nlp


def get_answer_pipeline(
    nlp,
    question_human,
    question_bot,
    context,
    use_cuda=True,
    verbose=True,
    is_persona=False,
):
    if verbose:
        print(f"question_human: {question_human}")
    answer_human = nlp(question=question_human, context=context)['answer']
    if verbose:
        print(nlp(question=question_human, context=context))
        print(f"answer_human: {answer_human}")
        print("~" * 50)
        print(f"question_bot: {question_bot}")
    answer_bot = nlp(question=question_bot, context=context)['answer']
    if verbose:

        print(f"answer_bot: {answer_bot}")

    return answer_human, answer_bot


def get_answer_unified_qa(
    nlp,
    question_human,
    question_bot,
    context,
    use_cuda=True,
    verbose=True,
    answer_bot_question_only=False,
    answer_human_question_only=False,
    device="cuda:0"
):
    # todo if_persona, only need to run once to speed things up
    model, tokenizer = nlp
    if not answer_bot_question_only:
        if verbose:
            print(f"question_human: {question_human}")
        input_string_human = utils.generate_document(context, question_human)
        answer_human = run_model(model, tokenizer, input_string_human, use_cuda, device=device)
        if verbose:
            print(f"answer_human: {answer_human}")
            print("~" * 50)

    if not answer_human_question_only:
        if verbose:
            print(f"question_bot: {question_bot}")
        input_string_bot = utils.generate_document(context, question_bot)
        answer_bot = run_model(model, tokenizer, input_string_bot, use_cuda, device=device)
        if verbose:
            print(f"answer_bot: {answer_bot}")

    if (not answer_bot_question_only) and (not answer_human_question_only):
        return answer_human, answer_bot
    else:
        if answer_bot_question_only:
            return answer_bot
        elif answer_human_question_only:
            return answer_human


def run_model(
    model, tokenizer, input_string, use_cuda=True, device="cuda:0", **generator_args
):
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    if torch.cuda.is_available() and use_cuda:
        input_ids = input_ids.to(device)
    res = model.generate(input_ids, **generator_args)
    answer = [tokenizer.decode(x) for x in res]
    if len(answer) == 1:
        return answer[0]
    else:
        raise ValueError(f"len(answer) > 1: {answer}")


def determine_no_answer(answer):
    """
    determine if an answer is [no_answer]
    """
    if answer in [utils.NO_ANSWER_TOKEN, '']:
        return True
    else:
        if "no" in answer.lower():
            return True
        else:
            return False


def is_same_answer(answer1, answer2):
    return str(answer1).lower() == str(answer2).lower()


def rephrase_question(question, who):
    if who == "human":
        question = question.replace("your", "the human's")
        question = question.replace("you", "the human")
    else:
        question = question.replace("your", "the bot's")
        question = question.replace("you", "the bot")
    return question


def detect_status_external_questions(
    answer_candidate, answer_context=None, answer_persona=None
):
    # import pdb

    # pdb.set_trace()
    status, status_persona = utils.GOOD, utils.GOOD
    if answer_context is not None:
        if determine_no_answer(answer_context):
            # the external question doesn't have an answer in the context
            status = utils.GOOD
        else:
            # the external question has an answer in the context
            if determine_no_answer(answer_candidate):
                # the external question doesn't have an answer in the candidate
                status = utils.GOOD
            else:
                # the external question has an answer in the candidate
                if is_same_answer(answer_context, answer_candidate):
                    # the answers from the context and the candidate are the same
                    status = utils.GOOD
                else:
                    # the answers from the context and the candidate are different
                    status = utils.CONTRADICTION

    if answer_persona is not None:
        if determine_no_answer(answer_persona):
            # the external question doesn't have an answer in the persona
            status_persona = utils.GOOD
        else:
            # the external question has an answer in the persona
            if determine_no_answer(answer_candidate):
                # the external question doesn't have an answer in the candidate
                status_persona = utils.GOOD
            else:
                # the external question has an answer in the candidate
                if is_same_answer(answer_persona, answer_candidate):
                    # the answers from the persona and the candidate are the same
                    status_persona = utils.GOOD
                else:
                    # the answers from the persona and the candidate are different
                    status_persona = utils.CONTRADICTION

    if status_persona in [utils.GOOD] and status in [utils.GOOD]:
        return utils.GOOD
    else:
        return utils.CONTRADICTION


def detect_status(
    answer_bot_persona,
    answer_human_context,
    answer_human_candidate,
    answer_bot_context,
    answer_bot_candidate,
    answer_human_next_response=None,
):
    if determine_no_answer(answer_human_context):
        # bot's question hasn't been asked before (doesn't have an answer in the context)
        status_human = utils.GOOD
    else:
        if answer_human_next_response is None:
            status_human = utils.REPETITIVE
        else:
            # bot's question has been asked before (has an answer in the context)
            if determine_no_answer(answer_human_next_response):
                # in the predicted next response, the human doesn't response to this question
                status_human = utils.GOOD
            else:
                # in the predicted next response, the human responses to this question
                if is_same_answer(answer_human_context, answer_human_next_response):
                    # in the predicted next response, the human responses respond with the same answer as in the context, repetition!
                    status_human = utils.REPETITIVE
                else:
                    # in the predicted next response, the human responses respond with a different answer as in the context, pass!
                    status_human = utils.GOOD

    if determine_no_answer(answer_bot_context):
        status_bot = utils.GOOD
    else:
        if determine_no_answer(answer_bot_candidate):
            status_bot = utils.GOOD
        else:
            if not is_same_answer(answer_bot_context, answer_bot_candidate):
                status_bot = utils.CONTRADICTION
            else:
                status_bot = utils.REPETITIVE

    # if bot answer is different from that in persona, also contradiction
    if status_bot == utils.GOOD:
        if determine_no_answer(answer_bot_persona):
            status_bot = utils.GOOD
        else:
            if determine_no_answer(answer_bot_candidate):
                status_bot = utils.GOOD
            else:
                if not is_same_answer(answer_bot_persona, answer_bot_candidate):
                    status_bot = utils.CONTRADICTION
                else:
                    status_bot = utils.GOOD

    if status_human == utils.GOOD and status_bot == utils.GOOD:
        return utils.GOOD
    elif status_human == utils.GOOD and status_bot == utils.REPETITIVE:
        return utils.REPETITIVE
    elif status_human == utils.GOOD and status_bot == utils.CONTRADICTION:
        return utils.CONTRADICTION

    elif status_human == utils.REPETITIVE and status_bot == utils.REPETITIVE:
        return utils.REPETITIVE
    elif status_human == utils.REPETITIVE and status_bot == utils.CONTRADICTION:
        return utils.CONTRADICTION
    elif status_human == utils.REPETITIVE and status_bot == utils.GOOD:
        return utils.REPETITIVE

    else:
        raise ValueError(
            f"shouldn't be another case: status_human: {status_human}, status_bot: {status_bot}"
        )


def answer_one_dialogue(
    get_answer,
    nlp,
    persona,
    context,
    candidate,
    future_human_response=None,
    verbose=True,
    generated_questions=[],
    device="cuda:0",
):
    if verbose:
        print(f"context: {context}")
        print("-" * 50)
        print(candidate)
        print("-" * 50)
    (
        answer_bot_persona,
        (answer_human_context, answer_bot_context),
        (answer_human_candidate, answer_bot_candidate),
    ) = ("", ("", ""), ("", ""))

    question_in_candidate = [
        s for s in nltk.sent_tokenize(candidate.replace("bot: ", "")) if s.endswith("?")
    ]
    qa_pair = {
        'human': {'context': {}, 'candidate': {}, 'future_response': {}},
        'bot': {'persona': {}, 'context': {}, 'candidate': {}},
        'generated_questions': {'context': {}, 'candidate': {}, 'persona': {}},
    }
    status_list = []
    for question in question_in_candidate:
        question_human = rephrase_question(question, "human")
        question_bot = rephrase_question(question, "bot")

        # context=persona
        if verbose:
            print(f"answer in persona: ")
            print()
        time0 = time.time()
        answer_bot_persona = get_answer(
            nlp=nlp,
            question_human=question_human,
            question_bot=question_bot,
            context=persona,
            verbose=verbose,
            answer_bot_question_only=True,
            device=device,
        )
        qa_pair['bot']['persona'][question_bot] = answer_bot_persona

        # context=context
        answer_human_context, answer_bot_context = get_answer(
            nlp=nlp,
            question_human=question_human,
            question_bot=question_bot,
            context=context,
            verbose=verbose,
            device=device,
        )
        if verbose:
            print("-" * 60)
            print(f"answer in context: ")
            print()
        qa_pair['bot']['context'][question_bot] = answer_bot_context
        qa_pair['human']['context'][question_human] = answer_human_context

        # context=candidate.replace(question, "")
        answer_human_candidate, answer_bot_candidate = get_answer(
            nlp=nlp,
            question_human=question_human,
            question_bot=question_bot,
            context=candidate.replace(question, ""),
            verbose=verbose,
            device=device,
        )
        qa_pair['bot']['candidate'][question_bot] = answer_bot_candidate
        qa_pair['human']['candidate'][question_human] = answer_human_candidate

        time1 = time.time()

        if verbose:
            print(f"took {time1-time0} seconds")

        # context=future_human_response
        if future_human_response is not None:
            answer_human_next_response = get_answer(
                nlp=nlp,
                question_human=question_human,
                question_bot=question_bot,
                context=future_human_response,
                verbose=verbose,
                answer_human_question_only=True,
                device=device,
            )
            qa_pair['human']['future_response'][
                question_human
            ] = answer_human_next_response
        else:
            answer_human_next_response = None
            qa_pair['human']['future_response'][question_human] = None

        status_temp = detect_status(
            answer_bot_persona,
            answer_human_context,
            answer_human_candidate,
            answer_bot_context,
            answer_bot_candidate,
            answer_human_next_response,
        )
        status_list.append(status_temp)

    for question in generated_questions:
        # context=context
        answer_context = get_answer(
            nlp=nlp,
            question_human=question,
            question_bot="",
            context=context,
            verbose=verbose,
            answer_bot_question_only=True,
            device=device,
        )
        answer_candidate = get_answer(
            nlp=nlp,
            question_human=question,
            question_bot="",
            context=candidate.replace(question, ""),
            verbose=verbose,
            answer_bot_question_only=True,
            device=device,
        )
        answer_persona = get_answer(
            nlp=nlp,
            question_human=question,
            question_bot="",
            context=persona,
            verbose=verbose,
            answer_bot_question_only=True,
            device=device,
        )

        if verbose:
            print("-" * 60)
            print(f"answer in candidate: ")
            print()

        qa_pair['generated_questions']['context'][question] = answer_context
        qa_pair['generated_questions']['candidate'][question] = answer_candidate
        qa_pair['generated_questions']['persona'][question] = answer_persona
        status_temp = detect_status_external_questions(
            answer_context=answer_context,
            answer_candidate=answer_candidate,
            answer_persona=answer_persona,
        )
        status_list.append(status_temp)

    if not ((utils.REPETITIVE in status_list) or (utils.CONTRADICTION in status_list)):
        status = utils.GOOD
    else:
        if utils.CONTRADICTION in status_list:
            status = utils.CONTRADICTION
        else:
            status = utils.REPETITIVE

    if verbose:
        print(f"status: {status}")
        print("\n\n\n")

    return (
        qa_pair,
        answer_bot_persona,
        (answer_human_context, answer_bot_context),
        (answer_human_candidate, answer_bot_candidate),
        status,
    )


def main(get_answer, nlp, personas, contexts, candidates):
    for persona, context, candidate in zip(personas, contexts, candidates):
        answer_one_dialogue(get_answer, nlp, persona, context, candidate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_unified_qa", type=bool, default=True)
    parser.add_argument(
        "-mp",
        "--model_path",
        type=str,
        default="~/ParlAI/data/models/t5/large/models_large_model.ckpt-1101200",
    )
    args = parser.parse_args()

    personas = ["", ""]
    contexts = [context1, context2]
    candidates = [candidate1, candidate2]
    if args.use_unified_qa:
        print("loading...")
        nlp = load_model_for_eval(args.model_path)
        main(get_answer_unified_qa, nlp, personas, contexts, candidates)
    else:
        nlp = pipeline("question-answering")
        main(get_answer_pipeline, nlp, personas, contexts, candidates)

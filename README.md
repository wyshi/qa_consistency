# qa_consistency

## Requirements

We use the [ParlAI](https://parl.ai/) framework for this project. You need to install ParlAI first.
The requirements are in requirements.txt

## Dialogue-based QA model
We build the QA model with the [DREAM dataset](https://github.com/nlpdata/dream).

## Question Generator
We build the question generator also with the [DREAM dataset](https://github.com/nlpdata/dream).

## Human evaluation
human_evaluation/ contains the scripts to build a [Blenderbot](https://parl.ai/projects/recipes/) using the classifiers. We also include a safety checker to prevent the chatbot from saying unsafe responses.

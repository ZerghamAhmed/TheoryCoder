# PRISM-Agent

python prism.py --query-mode "openai_direct" --multi-level --level-sets "{'demo_LEVELS': [8, 9, 10, 11, 12, 13]}" --learn-model

Run the experiment on all levels:

python prism.py --query-mode "openai_direct" --multi-level --level-sets "{'demo_LEVELS': [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]}" --experiment-dir run8 --learn-model

This repo's PRISM.py script builds off of tbrl.py 

https://github.com/c-j-bates/model-based-rl-with-llms.git

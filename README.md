This script allows you to enter prompts (separated by a &), model names and the model keywords in order to generate a grid comparing the outputs.
It works just like the regular x/y plot except it adds the token of the model to the beginning of each prompt and since it uses a & character as the separator you are able to make more complex prompts if you want since you can use commas and dots.

## How to use
The prompts goes in the top textbox, and they should be separated with a & characrter.
The middle field allows you to check which models you want to use. It is automatically generated from the list of models you have in your Automatic1111 UI models folder.
The bottom field is a list of all tokens* for your models. They should be entered in the same order as the models appears in your list. For model.ckpt you don't need to enter anything or just a whitespace
<img src="https://huggingface.co/Froddan/model_compare_script/resolve/main/instructions.png" />
*By tokens I mean the special word you should always use with a model, it's unique identifier or keyword.

## Example: While training mutiple models with different parameters I created a grid to compare the results using the same prompts and settings.
<img src="https://huggingface.co/Froddan/model_compare_script/resolve/main/xy_grid-0000-1085432319-.jpg" />



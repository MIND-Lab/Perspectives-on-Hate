# Modeling disagreement in transformer-based methods for hate speech detection
This GitHub repository contains the thesis work focused on applying contrastive learning to the context of disagreement.
The objective is to create a meaningful representation of the data in order to exploit the embeddings for the final prediction of disagreement.
The full explanation of the disagreement prediction process is shown in the thesis in the repository in pdf format.
## Run training process
```
main_supcon.py --batch_size 128 \
 --learning_rate 3e-5 \
 --temp 0.2 \
 --epochs 5
```
## Run final prediction
```
eval.py
```
## Final Model
The link of the trained model was uploaded on huggingface at the link https://huggingface.co/MF98/InfoNCE-HS

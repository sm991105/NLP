"""
author-gh: @adithya8
editor-gh: ykl7
"""

from tqdm import tqdm 

import torch

class Trainer:
    def __init__(self, model, ckpt_save_path, reverse_dictionary):
        self.model = model
        self.ckpt_save_path = ckpt_save_path
        self.reverse_dictionary = reverse_dictionary
    
    def training_step(self, center_word, context_word):
        loss =  self.model(center_word, context_word)
        return loss
    
    def train(self, dataset, max_training_steps, ckpt_steps, validation_words, device="cpu"):

        optim = torch.optim.SGD(self.model.parameters(), lr=1)
        self.model.to(device)
        self.model.train()
        self.losses = []
        
        t = tqdm(range(max_training_steps))
        for curr_step in t:
            optim.zero_grad()
            center_word, context_word = dataset.generate_batch()
            loss = self.training_step(center_word.to(device), context_word.to(device))
            loss.backward()
            optim.step()
            self.losses.append(loss.item())
            if curr_step: 
                t.set_description("Avg loss: %s"%(round(sum(self.losses[-2000:])/len(self.losses[-2000:]), 3)))
            if curr_step % 10000 == 0:
                self.model.print_closest(validation_words, self.reverse_dictionary)
            if curr_step%ckpt_steps == 0 and curr_step > 0: 
                self.save_ckpt(curr_step)

    def save_ckpt(self, curr_step):
        torch.save(self.model, "%s/%s.pt"%(self.ckpt_save_path, str(curr_step)))
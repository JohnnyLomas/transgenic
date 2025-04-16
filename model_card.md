---
library_name: transformers
tags: []
---

## Model Details

### Model Description

**TransGenic** is a transformer for DNA-to-annotation machine translation. Gene annotations specify the structure of a gene within a DNA sequence by providing the composition of each mRNA transcript based on the coordinate locations of sub-genic features, including coding sequences (CDS), introns, and unstranslated regions (UTR). TransGenic uses a HyenaDNA encoder with the Longformer decoder to predict a text-based annotation format from raw DNA sequence. This model is a 400M parameter checkpoint with a hidden size of 768, 12 layers, and 6 attention heads. 

This checkpoint was trained on 9 phylogenetically diverse plant species and acheived a maximum base-level F1 score of 92% in test set sequences from *Arabidopsis thaliana*.

## Uses

This model is intended to be used to generate *de novo* annotations for plant DNA sequences which contain a gene. The model can also be used to add alternatively spliced isoforms to known primary mRNA transcripts via prompt completion.

## How to Get Started with the Model

Example annotation of the AT1G58150.TAIR10 gene from *Arabidopsis*.
```
import torch
from transformers import AutoModel, AutoTokenizer

# Load the model and output tokenizer
model_name = "jlomas/HyenaTransgenic-768L12A6-400M"
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
gffTokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Load the HyenaDNA tokenizer
dnaTokenizer = AutoTokenizer.from_pretrained("LongSafari/hyenadna-large-1m-seqlen-hf", trust_remote_code=True)

# AT1G58150.TAIR10 gene sequence
seq = '''GCTTATGTTTATCTTTTGATCTGATCTATAAATATATATACAGGTTATCAAAAGGCCTCCACCAAAACCAACTCAACATCTCCGCCTCCATCTCCGCCTCCATCTCCGCCGCGAGTTCCAGACGCTCAAGAATTGGAGTACCTTAAATCCGACTCTTTTCCCGAACACGATGCGTAGAGTTGTCATTCGGACGGAGGTGTGCGTTCCGATAAAATTAGGCTACCGCCGCGGCTTTCAGACCTTCTAGAATTGGAGAAATTGTTTCCCGAACGCGAGGCGCTGAGTTGTCCTTTGGACGGAGATGAGGATTCCAATGAACTTAGGCTACGGCCGCTGGTTCCAGACGCTCAAGAATGGAAGTACCCTAAATCCAAGTTATTTCCCAGACACGCGGCGTGGAGTTGTCATTCGGGCGGAGGTGGAGGTGGAGGCGGTGGCCGTGTATTTACAAATAAAGTAAATGCGGTAGAAGAATTCAACTTAGGAGGACTGAAGGACAGCGAATCCGATTCCGATTCCGAGTAGGGAACTTTTAAAACAACTTTGATTATGGATTTCGATATCCAGAATAATTTTAATTCACTGCTGTTGGACTTGATTAATTTCCTATCACATAACGTTTTGGTTTAACTTTGTACGACCACCA'''

# Tokenize the input sequences and remove the [SEP] token
seqs = dnaTokenizer.batch_encode_plus(
	[seq], 
	return_tensors="pt")["input_ids"][:, :-1]

model.eval()
if torch.cuda.is_available():
    seqs = seqs.to("cuda")
    model.to("cuda")

# Prediction
outputs = model.generate(
	inputs=seqs,  
	num_return_sequences=1, 
	max_length=2048, 
	num_beams=2,
	do_sample=True,
	decoder_input_ids = None
)

# Convert to GSF
prediction = gffTokenizer.batch_decode(
    outputs.detach().cpu().numpy(), 
    skip_special_tokens=True
)

print(prediction)

# Output: ['<s>301|CDS1|523|+|A>CDS1']
```

## Training Details

### Training Data

Training was performed with nine plant genomes, including:
- *A. thaliana*
- *G. max* (Soybean)
- *O. sativa* (Rice)
- *S. bicolor* (Sorghum)
- *P. trichocarpa* (Poplar)
- *B. distachyon* (Grass)
- *V. vinifera* (Grape)
- *S. italica* (Millet)
- *P. patens* (Moss)

Gene-containing genomic sequences were randomly padded with neighboring sequence to produce input sequence lengths that were multiples of 6,144nt less than 49,152nt. Training was performed with the following parameters:

- Learning rate: 5e-5
- Effective batch size: 96
- Epochs: 22
- Loss function: Cross Entropy
- Mixed precision: BF16




## Citation [optional]


**BibTeX:**

[More Information Needed]

**APA:**

[More Information Needed]


## More Information [optional]

[More Information Needed]


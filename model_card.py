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

outputs = model.generate(
	inputs=seqs,  
	num_return_sequences=1, 
	max_length=2048, 
	num_beams=2,
	do_sample=True,
	decoder_input_ids = None
)

prediction = gffTokenizer.batch_decode(
    outputs.detach().cpu().numpy(), 
    skip_special_tokens=True
)

print(prediction)

# Output: ['<s>301|CDS1|523|+|A>CDS1']
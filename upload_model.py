from transformers import T5ForConditionalGeneration, T5Tokenizer

model = T5ForConditionalGeneration.from_pretrained("saved_summarization_model")
tokenizer = T5Tokenizer.from_pretrained("saved_summarization_model")

model.push_to_hub("JainishKumar12/text-summarizer-t5")
tokenizer.push_to_hub("JainishKumar12/text-summarizer-t5")
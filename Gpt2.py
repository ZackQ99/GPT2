import torch
from transformers import AutoTokenizer, AutoModelWithLMHead
import tkinter as tk
from sklearn.metrics import f1_score

# AI Name
AI_NAME = "GPT Bot"

# Custom UI
class ChatApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Chat")
        
        self.chat_log = tk.Text(self, height=30, width=50, state="disabled")
        self.chat_log.grid(row=0, column=0, padx=10, pady=10)
        
        self.entry_field = tk.Entry(self, width=50)
        self.entry_field.grid(row=1, column=0, padx=10, pady=10)
        self.entry_field.bind("<Return>", self.send_message)
        
        self.send_button = tk.Button(self, text="Send", command=self.send_message)
        self.send_button.grid(row=1, column=1, padx=10, pady=10)
        
    def send_message(self, event=None):
        user_input = self.entry_field.get()
        self.entry_field.delete(0, "end")
        response = get_response(user_input)
        self.display_message(user_input, is_user=True)
        self.display_message(response, is_user=False)
        
    def display_message(self, message, is_user=True):
        self.chat_log.configure(state="normal")
        if is_user:
            self.chat_log.insert("end", "You: " + message + "\n")
        else:
            self.chat_log.insert("end", AI_NAME + ": " + message + "\n")
        self.chat_log.configure(state="disabled")
        self.chat_log.see("end")


# AI Logic
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelWithLMHead.from_pretrained("gpt2")

def get_response(user_input):
    if not user_input:
        return "Sorry, I didn't catch that. Can you please repeat?"
    input_length = len(user_input.split())
    if input_length > 100:
        return "Sorry, your input is too long. Please keep it within 100 words."
    elif input_length > 20:
        return "Hmm, that's a complex question. Let me think for a moment..."
    else:
        input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
        response_ids = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
        return response

def evaluate_model():
    test_data = ["Hi there!", "What is your favorite color?", "Can you tell me a joke?", "What is the capital city of France?"]
    true_labels = ["Hi!", "I don't have favorite color, but I like all colors.", "Why don't scientists trust atoms? Because they make up everything!", "Paris"]
    predicted_labels = []
    for data in test_data:
        predicted_labels.append(get_response(data))
    score = f1_score(true_labels, predicted_labels, average='weighted')
    print("F1-score: {:.2f}".format(score))


# Main Program
if __name__ == "__main__":
    evaluate_model()
    app = ChatApp()
    app.mainloop()

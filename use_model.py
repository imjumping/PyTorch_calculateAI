import tkinter as tk
import torch
import torch.nn as nn
import numpy as np

# build NN
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(3, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# spawn a test question
def generate_single_question(max_num=100):
    a = np.random.randint(0, max_num)
    b = np.random.randint(0, max_num)
    op = np.random.choice([0, 1])  # 0 represents addition, 1 represents subtraction
    true_result = a + b if op == 0 else a - b
    op_symbol = '+' if op == 0 else '-'
    input_data = torch.tensor([[a, b, op]], dtype=torch.float32)
    return a, b, op_symbol, true_result, input_data

# INIT
model = SimpleNet()
model_path = 'trained_model.pth'
model.load_state_dict(torch.load(model_path))
model.eval()

# Global variables
input_data = None
true_result = None

# spawn question by user input
def new_question_user():
    global input_data, true_result
    try:
        # Get user input numbers
        a = int(entry_a.get())
        b = int(entry_b.get())
        op = 0  # Only addition
        true_result = a + b
        op_symbol = '+'
        input_data = torch.tensor([[a, b, op]], dtype=torch.float32)
        question_label.config(text=f"{a} {op_symbol} {b} = ?")
        result_label.config(text="")
    except ValueError:
        result_label.config(text="Please enter valid integers/请输入有效的整数")

# spawn question automatically
def new_question_auto():
    global input_data, true_result
    a, b, op_symbol, true_result, input_data = generate_single_question()
    question_label.config(text=f"{a} {op_symbol} {b} = ?")
    result_label.config(text="")

# main window
root = tk.Tk()
root.title("AI Math Quiz")

# input label
entry_a = tk.Entry(root, font=("Arial", 18))
entry_a.pack(side=tk.LEFT, padx=10)

plus_label = tk.Label(root, text="+", font=("Arial", 18))
plus_label.pack(side=tk.LEFT)

entry_b = tk.Entry(root, font=("Arial", 18))
entry_b.pack(side=tk.LEFT, padx=10)

equal_label = tk.Label(root, text="=", font=("Arial", 18))
equal_label.pack(side=tk.LEFT)

# view question
question_label = tk.Label(root, text="", font=("Arial", 24))
question_label.pack(pady=20)

# Button to generate question by user input
new_question_user_label = tk.Label(root, text="手动输入生成题目👇", font=("Arial", 12))
new_question_user_label.pack(pady=5)
new_question_user_button = tk.Button(root, text="input question", command=new_question_user)
new_question_user_button.pack(pady=5)

# Button to generate question automatically
new_question_auto_label = tk.Label(root, text="自动生成题目👇", font=("Arial", 12))
new_question_auto_label.pack(pady=5)
new_question_auto_button = tk.Button(root, text="auto spawm question", command=new_question_auto)
new_question_auto_button.pack(pady=5)

# upload button
def submit_answer():
    with torch.no_grad():
        prediction = model(input_data).item()
    result_label.config(text=f"正确答案/True's: {true_result}, 预测答案/NN's: {prediction:.0f}")

submit_label = tk.Label(root, text="预测答案👇", font=("Arial", 12))
submit_label.pack(pady=5)
submit_button = tk.Button(root, text="Start AI prediction", command=submit_answer)
submit_button.pack(pady=5)

# view output
result_label = tk.Label(root, text="", font=("Arial", 18))
result_label.pack(pady=20)

# run
root.mainloop()
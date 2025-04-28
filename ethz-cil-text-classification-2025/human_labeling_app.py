import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.metrics import mean_absolute_error


# Just run this file and then label all sentences
# In the end the results will be saved in a csv and the L-score will be printed

full_df = pd.read_csv("data/training.csv")
label_mapping = {'negative': -1, 'neutral': 0, 'positive': 1}

# select how many samples for labeling
df = full_df.sample(n=500, random_state=42).copy().reset_index(drop=True)
df = df[["id", "sentence", "label"]]
df["human_label"] = ""

class LabelingApp:
    def __init__(self, master, df):
        self.master = master
        self.df = df
        self.index = 0

        master.title("Sentiment Labeling Tool")
        master.geometry("800x600")
        master.configure(bg="white")
        master.resizable(False, False)

        self.label = tk.Label(master, text="", wraplength=600, font=("Helvetica", 20), justify="left", bg="white", fg="black")
        self.label.pack(pady=20)

        self.btn_frame = tk.Frame(master)
        self.btn_frame.pack()
        self.btn_frame.configure(bg="white")

        self.negative_btn = tk.Button(self.btn_frame, text="Negative", command=lambda: self.save_label("negative"), bg="red", fg="white", font=("Helvetica", 12, "bold"))
        self.negative_btn.grid(row=0, column=0, padx=10)

        self.neutral_btn = tk.Button(self.btn_frame, text="Neutral", command=lambda: self.save_label("neutral"), font=("Helvetica", 12, "bold"))
        self.neutral_btn.grid(row=0, column=1, padx=10)

        self.positive_btn = tk.Button(self.btn_frame, text="Positive", command=lambda: self.save_label("positive"), bg="green", fg="white", font=("Helvetica", 12, "bold"))
        self.positive_btn.grid(row=0, column=2, padx=10)

        self.update_sentence()

    def update_sentence(self):
        if self.index < len(self.df):
            sentence = self.df.iloc[self.index]["sentence"]
            self.label.config(text=f"[{self.index+1}/{len(self.df)}] {sentence}")
        else:
            self.finish_labeling()

    def save_label(self, label):
        self.df.at[self.index, "human_label"] = label
        self.index += 1
        self.update_sentence()

    def finish_labeling(self):
        self.label.config(text="Done! Saving file and calculating score...")

        # save labeled file
        self.df.to_csv("data/human_label_sample_labeled.csv", index=False)

        df_eval = self.df.copy()
        df_eval["label_encoded"] = df_eval["label"].map(label_mapping)
        df_eval["human_label_encoded"] = df_eval["human_label"].map(label_mapping)

        y_true = df_eval["label_encoded"]
        y_human = df_eval["human_label_encoded"]

        mae = mean_absolute_error(y_true, y_human)
        L_score = 0.5 * (2 - mae)

        print(f"Human vs True Labels — L-Score: {L_score:.4f}")

        messagebox.showinfo("Done", f"Labels saved.\nL-Score: {L_score:.4f}")
        self.master.quit()


root = tk.Tk()
app = LabelingApp(root, df)
root.mainloop()

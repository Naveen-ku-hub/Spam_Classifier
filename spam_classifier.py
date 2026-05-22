import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# 1. Create a Built-in Dataset (No external file downloads required!)
# A simple collection of sample emails labeled as 0 (Ham/Safe) or 1 (Spam)
emails = [
    "Hey, are we still meeting for lunch today at 1 PM?",  # 0
    "URGENT! Your credit card has been compromised. Click here immediately!!",  # 1
    "Can you please send me the final report by end of day?",  # 0
    "WINNER! You have won a free $1000 Walmart gift card! Call now to claim.",  # 1
    "Don't forget to bring the documents for the meeting tomorrow.",  # 0
    "CONGRATULATIONS! You are selected for a free tropical vacation trip!",  # 1
    "Thanks for the update, I will look into this tomorrow morning.",  # 0
    "LOSE WEIGHT FAST! Secrets the doctors don't want you to know!!",  # 1
    "Hi Mom, just calling to check in on how you are feeling.",  # 0
    "Double your income working from home! Only a few spots left, sign up!",  # 1
]

# Labels: 0 = Ham (Safe Email), 1 = Spam
labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

print("Step 1: Preparing training dataset...")

# 2. Convert text data into numbers (Vectorization)
# Machine Learning models can't read text directly, so we convert words to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)
y = np.array(labels)

# 3. Train the Naive Bayes Classifier model
print("Step 2: Training the Naive Bayes model...")
model = MultinomialNB()
model.fit(X, y)
print("Model training completed successfully!\n")


# 4. Function to predict new emails
def classify_email(text_content):
    # Convert the input text using the same vectorizer
    input_vectorized = vectorizer.transform([text_content])
    # Make the prediction
    prediction = model.predict(input_vectorized)

    if prediction[0] == 1:
        return "🚨 SPAM 🚨"
    else:
        return "✅ HAM (Safe Email) ✅"


# 5. Test the model with brand new, unseen text
if __name__ == "__main__":
    print("--- Testing the Spam Classifier ---")

    test_email_1 = "Hey friend, let's grab some coffee this weekend."
    test_email_2 = (
        "CASH PRIZE!! Click this link to get your inheritance money right now!"
    )

    print(f"Email 1: '{test_email_1}'")
    print(f"Prediction: {classify_email(test_email_1)}\n")

    print(f"Email 2: '{test_email_2}'")
    print(f"Prediction: {classify_email(test_email_2)}\n")

    # Interactive loop so you can test your own emails in the terminal
    print("-----------------------------------")
    print("Type your own custom email text to test it live:")
    user_input = input("Enter email text: ")
    print(f"Prediction for your input: {classify_email(user_input)}")
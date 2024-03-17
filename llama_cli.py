from llama_cpp import Llama, llama_types
from sys import stderr

MODEL_PATH = "/to_path_pou_evales_to_llama/llama-2-7b-chat.Q5_K_M.gguf"
model = Llama(
      model_path=MODEL_PATH,
      # n_gpu_layers=-1, # Κάνε uncomment για να χρησιμοποιήσεις την GPU.
      chat_format="llama-2",
      # Το context window. 0 = χρησιμοποίησε όσο όρίζει το model
      # Όσο μεγαλύτερο, τόσα περισσότερα "θυμάται" το μοντέλο,
      # αλλά αυξάνει και τον χρόνο εκτέλεσης...
      n_ctx=0  
)

chat_history: list[llama_types.ChatCompletionRequestMessage] = []

def get_llm_output(chat_history):
  response = model.create_chat_completion(
    messages=chat_history,
    # max_new_tokens=150  # Κάνε uncomment για να περιορίσεις το μέγεθος των απαντήσεων
  )

  chat_history.append(response["choices"][0]["message"])
  return chat_history

def get_user_input() -> str:
  while True:
    user_input = input("> ")
    if user_input.strip():
      return user_input

    print("Chat message empty. Please retry.", file=stderr)

if __name__ == "__main__":
  # Το πρώτο message μπορεί να είναι τύπου system.
  # Πχ. "You're a scientist"
  # Αυτό ρυθμίζει την συμπεριφορά του μοντέλου.
  system_msg = input("SYS (leave empty to not use a system message)> ")
  if system_msg.strip():
    chat_history.append({
      "role": "system",
      "content": system_msg
    })

  while True:
    try:
      user_msg = chat_history.append({
        "role": "user",
        "content": get_user_input()
      })
      
      chat_history = get_llm_output(chat_history)
      print("AI> " + chat_history[-1]["content"])
    except KeyboardInterrupt:
      print("Exiting...", file=stderr)
      exit(0)

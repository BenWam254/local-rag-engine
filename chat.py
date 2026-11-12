import requests
import sys

def main():
    print("==================================================")
    print("🤖 Forge Local RAG Engine - Interactive Chat")
    print("==================================================")
    print("Type 'exit' or 'quit' to stop.\n")
    
    url = "http://127.0.0.1:8000/api/chat"
    
    while True:
        try:
            user_input = input("👤 You: ")
            
            if user_input.lower() in ['exit', 'quit']:
                print("Shutting down chat interface...")
                break
            if not user_input.strip():
                continue
            
            print("🧠 Searching vault & thinking...")
            
            # Robust network request with extended timeout for local LLM generation
            response = requests.post(url, json={"query": user_input}, timeout=120)
            response.raise_for_status()
            payload = response.json()
            
            # Extract and print the LLM response
            answer = payload.get("response", "No response found.")
            print(f"\n🤖 Assistant: {answer}")
            
            # Extract and print the local file citations
            sources = payload.get("sources", [])
            if sources:
                print("\n📚 Sources:")
                for src in sources:
                    print(f"  🔗 {src}")
            print("\n" + "-"*50 + "\n")
            
        except requests.RequestException as exc:
            print(f"⚠️ Error contacting the local RAG service: {exc}")
        except ValueError:
            print("⚠️ The server returned an invalid response.")
        except KeyboardInterrupt:
            print("\nShutting down chat interface...")
            break
        except Exception as exc:
            print(f"⚠️ Unexpected error: {exc}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nShutting down chat interface...")
        sys.exit(0)
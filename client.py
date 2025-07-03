import requests
import time

class MCPClient:
    def __init__(self, base_url="http://localhost:7860"):
        self.base_url = base_url
    
    def send_query(self, query: str):
        """Send query to MCP server"""
        try:
            response = requests.post(
                f"{self.base_url}/api/query",
                json={"query": query},
                timeout=30
            )
            return response.json()["response"]
        except Exception as e:
            return f"Server Error: {str(e)}"
    
    def test_ollama(self, query: str, model="gemma:2b"):
        """Test with local Ollama"""
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": query,
                    "stream": False
                },
                timeout=15
            )
            return response.json()["response"]
        except Exception as e:
            return f"Ollama Error: {str(e)}"

if __name__ == "__main__":
    client = MCPClient()
    
    print("MCP Client - Type 'exit' to quit")
    print("Available modes:\n1. MCP Server\n2. Ollama\n3. Both")
    
    while True:
        mode = input("\nSelect mode (1/2/3): ").strip()
        if mode not in ["1", "2", "3"]:
            print("Invalid mode selection")
            continue
            
        message = input("\nYou: ")
        if message.lower() in ["exit", "quit"]:
            break
        
        if mode in ["1", "3"]:
            print("\n[MCP Server]")
            start = time.time()
            response = client.send_query(message)
            print(f"Response ({time.time()-start:.2f}s): {response}")
        
        if mode in ["2", "3"]:
            print("\n[Ollama]")
            start = time.time()
            response = client.test_ollama(message)
            print(f"Response ({time.time()-start:.2f}s): {response}")
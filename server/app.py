import os
import subprocess
import sys
import time
import requests

def main():
    # Detect the directory of this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"Starting NetPulse Node.js server from {current_dir}...")
    
    # Ensure node_modules are installed if they aren't
    if not os.path.exists(os.path.join(current_dir, "node_modules")):
        print("node_modules not found, attempting npm install...")
        try:
            subprocess.run(["npm", "install"], cwd=current_dir, check=True)
        except Exception as e:
            print(f"Failed to run npm install: {e}")
            sys.exit(1)

    # Start the Node.js server
    # We use 'npm start' which should run the server on port 7860
    try:
        # We start it as a subprocess
        process = subprocess.Popen(["npm", "start"], cwd=current_dir)
        
        # Give it a moment to start
        print("Waiting for server to be healthy...")
        max_retries = 30
        for i in range(max_retries):
            try:
                # Assuming the server has a health endpoint as per openenv.yaml
                response = requests.get("http://localhost:7860/health", timeout=1)
                if response.status_code == 200:
                    print("NetPulse server is healthy and running on port 7860.")
                    break
            except Exception:
                pass
            time.sleep(1)
        else:
            print("Warning: Server did not respond to health check after 30 seconds.")

        # Keep the Python process alive while the server is running
        process.wait()
        
    except KeyboardInterrupt:
        print("Shutting down...")
        process.terminate()
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

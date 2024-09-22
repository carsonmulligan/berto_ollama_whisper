# create python 3.11 conda env with name vllm
conda create -n vllm python=3.11
conda activate vllm

# Install vLLM from pip:
pip install vllm

# Load and run the model:
vllm serve "clu-ling/whisper-small-spanish"

# Call the server using curl:
curl -X POST "http://localhost:8000/v1/chat/completions" \ 
	-H "Content-Type: application/json" \ 
	--data '{
		"model": "clu-ling/whisper-small-spanish"
		"messages": [
			{"role": "user", "content": "Hello!"}
		]
	}'
# create python 3.11 conda env with name vllm
conda create -n vllm python=3.11
conda activate vllm

# Install vLLM from pip:
conda install vllm

# Load and run the model:
vllm serve "clu-ling/whisper-small-spanish"

# Call the server using curl:
curl -X POST "http://localhost:8000/v1/chat/completions" \ 
	-H "Content-Type: application/json" \ 
	--data '{
		"model": "clu-ling/whisper-small-spanish"
		"messages": [
			{"role": "user", "content": "Hola Chamo, soy Don Berto de la Isla de Pascua, Cómo va la vida, mi compa? Que tal tu día? tu vaya bien, mano?"}
		]
	}'
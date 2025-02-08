from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import ChatRequestSerializer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Loading the model and tokenizer once at startup
MODEL_NAME = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

class ChatAPIView(APIView):
    def post(self, request):
        serializer = ChatRequestSerializer(data=request.data)
        if serializer.is_valid():
            prompt = serializer.validated_data['prompt']

            # Tokenize the prompt
            inputs = tokenizer.encode(prompt, return_tensors="pt")
            # Generate the response
            outputs = model.generate(
                inputs,
                max_length=1000,
                do_sample=True,
                top_p=0.95,
                top_k=50,
                pad_token_id=tokenizer.eos_token_id
            )
            response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            return Response({'response': response_text})
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

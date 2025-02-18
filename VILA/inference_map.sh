python -W ignore llava/eval/run_vila.py \
    --model-path Efficient-Large-Model/Llama-3-VILA1.5-8b \
    --conv-mode llama_3 \
    --query "<image>\n Please describe this image in a structured way." \
    --image-file "Houthi_CoT_in_Yemen_February_19_2024_1.png"

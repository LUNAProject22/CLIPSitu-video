    python -W ignore ./llava/eval/VILA_imsitu.py \
    --model-path Efficient-Large-Model/Llama-3-VILA1.5-8b \
    --data_file ./test_empty.json \
    --image_dir ../data/of500_images_resized \
    --sample_data -1 \
    --output_file ./imsitu_output_top1_test_empty.json \
    --json_visual_prompt_file json_visual_prompt.txt \
    --example_images ../data/of500_images_resized/prowling_234.jpg,../data/of500_images_resized/spraying_141.jpg,../data/of500_images_resized/slipping_166.jpg \
    --conv-mode llama_3
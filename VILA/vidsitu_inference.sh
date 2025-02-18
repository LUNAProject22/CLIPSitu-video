    python -W ignore ./llava/eval/VILA_vidsitu.py \
    --model-path Efficient-Large-Model/Llama-3-VILA1.5-8b \
    --data_file /home/dhruv/Projects/VidSitu/vidsitu_data/vidsitu_annotations/vidsitu_valid_subset_GT.json \
    --image_dir /home/dhruv/Projects/VidSitu/vidsitu_data/vsitu_11_frames_per_vid \
    --output_file ./vidsitu_output_top1_valid_update.json \
    --vidsitu_visual_prompt_file vidsitu_visual_prompt2.txt \
    --example_images /home/dhruv/Projects/VidSitu/vidsitu_data/vsitu_11_frames_per_vid/v_V-vgoh3ukPc_seg_95_105/frame_06.jpg,/home/dhruv/Projects/VidSitu/vidsitu_data/vsitu_11_frames_per_vid/v_Kz234-_khjs_seg_75_85/frame_09.jpg,/home/dhruv/Projects/VidSitu/vidsitu_data/vsitu_11_frames_per_vid/v_8wDlH8jWxYk_seg_5_15/frame_03.jpg,/home/dhruv/Projects/VidSitu/vidsitu_data/vsitu_11_frames_per_vid/v_rtqgJvhrswY_seg_80_90/frame_07.jpg,/home/dhruv/Projects/VidSitu/vidsitu_data/vsitu_11_frames_per_vid/v_tSZtoveaa0A_seg_40_50/frame_01.jpg \
    --conv-mode llama_3
# ###############################################################
# ## Set A
# ###############################################################

# ##### Train c0 #####

# # # # Train: LR; c0                     Test: LR and HR; c0;
# python test.py --model_id 0 --resolution_id 0 --compression_id 0 --test_resolution_id 0 --test_compression_id 0
# python test.py --model_id 0 --resolution_id 0 --compression_id 0 --test_resolution_id 1 --test_compression_id 0
# python test.py --model_id 1 --resolution_id 0 --compression_id 0 --test_resolution_id 0 --test_compression_id 0
# python test.py --model_id 1 --resolution_id 0 --compression_id 0 --test_resolution_id 1 --test_compression_id 0
# python test.py --model_id 2 --resolution_id 0 --compression_id 0 --test_resolution_id 0 --test_compression_id 0
# python test.py --model_id 2 --resolution_id 0 --compression_id 0 --test_resolution_id 1 --test_compression_id 0
# python test.py --model_id 3 --resolution_id 0 --compression_id 0 --test_resolution_id 0 --test_compression_id 0
# python test.py --model_id 3 --resolution_id 0 --compression_id 0 --test_resolution_id 1 --test_compression_id 0
# python test.py --model_id 4 --resolution_id 0 --compression_id 0 --test_resolution_id 0 --test_compression_id 0
# python test.py --model_id 4 --resolution_id 0 --compression_id 0 --test_resolution_id 1 --test_compression_id 0
# python test.py --model_id 5 --resolution_id 0 --compression_id 0 --test_resolution_id 0 --test_compression_id 0
# python test.py --model_id 5 --resolution_id 0 --compression_id 0 --test_resolution_id 1 --test_compression_id 0

# # # Train: HR; c0                     Test: LR and HR; c0;
# python test.py --model_id 0 --resolution_id 1 --compression_id 0 --test_resolution_id 0 --test_compression_id 0
# python test.py --model_id 0 --resolution_id 1 --compression_id 0 --test_resolution_id 1 --test_compression_id 0
# python test.py --model_id 1 --resolution_id 1 --compression_id 0 --test_resolution_id 0 --test_compression_id 0
# python test.py --model_id 1 --resolution_id 1 --compression_id 0 --test_resolution_id 1 --test_compression_id 0
# python test.py --model_id 2 --resolution_id 1 --compression_id 0 --test_resolution_id 0 --test_compression_id 0
# python test.py --model_id 2 --resolution_id 1 --compression_id 0 --test_resolution_id 1 --test_compression_id 0
# python test.py --model_id 3 --resolution_id 1 --compression_id 0 --test_resolution_id 0 --test_compression_id 0
# python test.py --model_id 3 --resolution_id 1 --compression_id 0 --test_resolution_id 1 --test_compression_id 0
# python test.py --model_id 4 --resolution_id 1 --compression_id 0 --test_resolution_id 0 --test_compression_id 0
# python test.py --model_id 4 --resolution_id 1 --compression_id 0 --test_resolution_id 1 --test_compression_id 0
# python test.py --model_id 5 --resolution_id 1 --compression_id 0 --test_resolution_id 0 --test_compression_id 0
# python test.py --model_id 5 --resolution_id 1 --compression_id 0 --test_resolution_id 1 --test_compression_id 0

# # # Train: LR; c0                     Test: LR and HR; c23;
# python test.py --model_id 0 --resolution_id 0 --compression_id 0 --test_resolution_id 0 --test_compression_id 1
# python test.py --model_id 0 --resolution_id 0 --compression_id 0 --test_resolution_id 1 --test_compression_id 1
# python test.py --model_id 1 --resolution_id 0 --compression_id 0 --test_resolution_id 0 --test_compression_id 1
# python test.py --model_id 1 --resolution_id 0 --compression_id 0 --test_resolution_id 1 --test_compression_id 1
# python test.py --model_id 2 --resolution_id 0 --compression_id 0 --test_resolution_id 0 --test_compression_id 1
# python test.py --model_id 2 --resolution_id 0 --compression_id 0 --test_resolution_id 1 --test_compression_id 1
# python test.py --model_id 3 --resolution_id 0 --compression_id 0 --test_resolution_id 0 --test_compression_id 1
# python test.py --model_id 3 --resolution_id 0 --compression_id 0 --test_resolution_id 1 --test_compression_id 1
# python test.py --model_id 4 --resolution_id 0 --compression_id 0 --test_resolution_id 0 --test_compression_id 1
# python test.py --model_id 4 --resolution_id 0 --compression_id 0 --test_resolution_id 1 --test_compression_id 1
# python test.py --model_id 5 --resolution_id 0 --compression_id 0 --test_resolution_id 0 --test_compression_id 1
# python test.py --model_id 5 --resolution_id 0 --compression_id 0 --test_resolution_id 1 --test_compression_id 1

# # # Train: HR; c0                     Test: LR and HR; c23;
# python test.py --model_id 0 --resolution_id 1 --compression_id 0 --test_resolution_id 0 --test_compression_id 1
# python test.py --model_id 0 --resolution_id 1 --compression_id 0 --test_resolution_id 1 --test_compression_id 1
# python test.py --model_id 1 --resolution_id 1 --compression_id 0 --test_resolution_id 0 --test_compression_id 1
# python test.py --model_id 1 --resolution_id 1 --compression_id 0 --test_resolution_id 1 --test_compression_id 1
# python test.py --model_id 2 --resolution_id 1 --compression_id 0 --test_resolution_id 0 --test_compression_id 1
# python test.py --model_id 2 --resolution_id 1 --compression_id 0 --test_resolution_id 1 --test_compression_id 1
# python test.py --model_id 3 --resolution_id 1 --compression_id 0 --test_resolution_id 0 --test_compression_id 1
# python test.py --model_id 3 --resolution_id 1 --compression_id 0 --test_resolution_id 1 --test_compression_id 1
# python test.py --model_id 4 --resolution_id 1 --compression_id 0 --test_resolution_id 0 --test_compression_id 1
# python test.py --model_id 4 --resolution_id 1 --compression_id 0 --test_resolution_id 1 --test_compression_id 1
# python test.py --model_id 5 --resolution_id 1 --compression_id 0 --test_resolution_id 0 --test_compression_id 1
# python test.py --model_id 5 --resolution_id 1 --compression_id 0 --test_resolution_id 1 --test_compression_id 1

# # # Train: LR; c0                     Test: LR and HR; c40;
# python test.py --model_id 0 --resolution_id 0 --compression_id 0 --test_resolution_id 0 --test_compression_id 2
# python test.py --model_id 0 --resolution_id 0 --compression_id 0 --test_resolution_id 1 --test_compression_id 2
# python test.py --model_id 1 --resolution_id 0 --compression_id 0 --test_resolution_id 0 --test_compression_id 2
# python test.py --model_id 1 --resolution_id 0 --compression_id 0 --test_resolution_id 1 --test_compression_id 2
# python test.py --model_id 2 --resolution_id 0 --compression_id 0 --test_resolution_id 0 --test_compression_id 2
# python test.py --model_id 2 --resolution_id 0 --compression_id 0 --test_resolution_id 1 --test_compression_id 2
# python test.py --model_id 3 --resolution_id 0 --compression_id 0 --test_resolution_id 0 --test_compression_id 2
# python test.py --model_id 3 --resolution_id 0 --compression_id 0 --test_resolution_id 1 --test_compression_id 2
# python test.py --model_id 4 --resolution_id 0 --compression_id 0 --test_resolution_id 0 --test_compression_id 2
# python test.py --model_id 4 --resolution_id 0 --compression_id 0 --test_resolution_id 1 --test_compression_id 2
# python test.py --model_id 5 --resolution_id 0 --compression_id 0 --test_resolution_id 0 --test_compression_id 2
# python test.py --model_id 5 --resolution_id 0 --compression_id 0 --test_resolution_id 1 --test_compression_id 2

# # # Train: HR; c0                     Test: LR and HR; c40;
# python test.py --model_id 0 --resolution_id 1 --compression_id 0 --test_resolution_id 0 --test_compression_id 2
# python test.py --model_id 0 --resolution_id 1 --compression_id 0 --test_resolution_id 1 --test_compression_id 2
# python test.py --model_id 1 --resolution_id 1 --compression_id 0 --test_resolution_id 0 --test_compression_id 2
# python test.py --model_id 1 --resolution_id 1 --compression_id 0 --test_resolution_id 1 --test_compression_id 2
# python test.py --model_id 2 --resolution_id 1 --compression_id 0 --test_resolution_id 0 --test_compression_id 2
# python test.py --model_id 2 --resolution_id 1 --compression_id 0 --test_resolution_id 1 --test_compression_id 2
# python test.py --model_id 3 --resolution_id 1 --compression_id 0 --test_resolution_id 0 --test_compression_id 2
# python test.py --model_id 3 --resolution_id 1 --compression_id 0 --test_resolution_id 1 --test_compression_id 2
# python test.py --model_id 4 --resolution_id 1 --compression_id 0 --test_resolution_id 0 --test_compression_id 2
# python test.py --model_id 4 --resolution_id 1 --compression_id 0 --test_resolution_id 1 --test_compression_id 2
# python test.py --model_id 5 --resolution_id 1 --compression_id 0 --test_resolution_id 0 --test_compression_id 2
# python test.py --model_id 5 --resolution_id 1 --compression_id 0 --test_resolution_id 1 --test_compression_id 2

# ###################################################################

# ##### Train c23 #####

# # # Train: LR; c23                     Test: LR and HR; c0;
# python test.py --model_id 0 --resolution_id 0 --compression_id 1 --test_resolution_id 0 --test_compression_id 0
# python test.py --model_id 0 --resolution_id 0 --compression_id 1 --test_resolution_id 1 --test_compression_id 0
# python test.py --model_id 1 --resolution_id 0 --compression_id 1 --test_resolution_id 0 --test_compression_id 0
# python test.py --model_id 1 --resolution_id 0 --compression_id 1 --test_resolution_id 1 --test_compression_id 0
# python test.py --model_id 2 --resolution_id 0 --compression_id 1 --test_resolution_id 0 --test_compression_id 0
# python test.py --model_id 2 --resolution_id 0 --compression_id 1 --test_resolution_id 1 --test_compression_id 0
# python test.py --model_id 3 --resolution_id 0 --compression_id 1 --test_resolution_id 0 --test_compression_id 0
# python test.py --model_id 3 --resolution_id 0 --compression_id 1 --test_resolution_id 1 --test_compression_id 0
# python test.py --model_id 4 --resolution_id 0 --compression_id 1 --test_resolution_id 0 --test_compression_id 0
# python test.py --model_id 4 --resolution_id 0 --compression_id 1 --test_resolution_id 1 --test_compression_id 0
# python test.py --model_id 5 --resolution_id 0 --compression_id 1 --test_resolution_id 0 --test_compression_id 0
# python test.py --model_id 5 --resolution_id 0 --compression_id 1 --test_resolution_id 1 --test_compression_id 0


# # # Train: HR; c23                     Test: LR and HR; c0;
# python test.py --model_id 0 --resolution_id 1 --compression_id 1 --test_resolution_id 0 --test_compression_id 0
# python test.py --model_id 0 --resolution_id 1 --compression_id 1 --test_resolution_id 1 --test_compression_id 0
# python test.py --model_id 1 --resolution_id 1 --compression_id 1 --test_resolution_id 0 --test_compression_id 0
# python test.py --model_id 1 --resolution_id 1 --compression_id 1 --test_resolution_id 1 --test_compression_id 0
# python test.py --model_id 2 --resolution_id 1 --compression_id 1 --test_resolution_id 0 --test_compression_id 0
# python test.py --model_id 2 --resolution_id 1 --compression_id 1 --test_resolution_id 1 --test_compression_id 0
# python test.py --model_id 3 --resolution_id 1 --compression_id 1 --test_resolution_id 0 --test_compression_id 0
# python test.py --model_id 3 --resolution_id 1 --compression_id 1 --test_resolution_id 1 --test_compression_id 0
# python test.py --model_id 4 --resolution_id 1 --compression_id 1 --test_resolution_id 0 --test_compression_id 0
# python test.py --model_id 4 --resolution_id 1 --compression_id 1 --test_resolution_id 1 --test_compression_id 0
# python test.py --model_id 5 --resolution_id 1 --compression_id 1 --test_resolution_id 0 --test_compression_id 0
# python test.py --model_id 5 --resolution_id 1 --compression_id 1 --test_resolution_id 1 --test_compression_id 0

# # # Train: LR; c23                     Test: LR and HR; c23;
# python test.py --model_id 0 --resolution_id 0 --compression_id 1 --test_resolution_id 0 --test_compression_id 1
# python test.py --model_id 0 --resolution_id 0 --compression_id 1 --test_resolution_id 1 --test_compression_id 1
# python test.py --model_id 1 --resolution_id 0 --compression_id 1 --test_resolution_id 0 --test_compression_id 1
# python test.py --model_id 1 --resolution_id 0 --compression_id 1 --test_resolution_id 1 --test_compression_id 1
# python test.py --model_id 2 --resolution_id 0 --compression_id 1 --test_resolution_id 0 --test_compression_id 1
# python test.py --model_id 2 --resolution_id 0 --compression_id 1 --test_resolution_id 1 --test_compression_id 1
# python test.py --model_id 3 --resolution_id 0 --compression_id 1 --test_resolution_id 0 --test_compression_id 1
# python test.py --model_id 3 --resolution_id 0 --compression_id 1 --test_resolution_id 1 --test_compression_id 1
# python test.py --model_id 4 --resolution_id 0 --compression_id 1 --test_resolution_id 0 --test_compression_id 1
# python test.py --model_id 4 --resolution_id 0 --compression_id 1 --test_resolution_id 1 --test_compression_id 1
# python test.py --model_id 5 --resolution_id 0 --compression_id 1 --test_resolution_id 0 --test_compression_id 1
# python test.py --model_id 5 --resolution_id 0 --compression_id 1 --test_resolution_id 1 --test_compression_id 1

# # # Train: HR; c23                     Test: LR and HR; c23;
# python test.py --model_id 0 --resolution_id 1 --compression_id 1 --test_resolution_id 0 --test_compression_id 1
# python test.py --model_id 0 --resolution_id 1 --compression_id 1 --test_resolution_id 1 --test_compression_id 1
# python test.py --model_id 1 --resolution_id 1 --compression_id 1 --test_resolution_id 0 --test_compression_id 1
# python test.py --model_id 1 --resolution_id 1 --compression_id 1 --test_resolution_id 1 --test_compression_id 1
# python test.py --model_id 2 --resolution_id 1 --compression_id 1 --test_resolution_id 0 --test_compression_id 1
# python test.py --model_id 2 --resolution_id 1 --compression_id 1 --test_resolution_id 1 --test_compression_id 1
# python test.py --model_id 3 --resolution_id 1 --compression_id 1 --test_resolution_id 0 --test_compression_id 1
# python test.py --model_id 3 --resolution_id 1 --compression_id 1 --test_resolution_id 1 --test_compression_id 1
# python test.py --model_id 4 --resolution_id 1 --compression_id 1 --test_resolution_id 0 --test_compression_id 1
# python test.py --model_id 4 --resolution_id 1 --compression_id 1 --test_resolution_id 1 --test_compression_id 1
# python test.py --model_id 5 --resolution_id 1 --compression_id 1 --test_resolution_id 0 --test_compression_id 1
# python test.py --model_id 5 --resolution_id 1 --compression_id 1 --test_resolution_id 1 --test_compression_id 1

# # # Train: LR; c23                     Test: LR and HR; c40;
# python test.py --model_id 0 --resolution_id 0 --compression_id 1 --test_resolution_id 0 --test_compression_id 2
# python test.py --model_id 0 --resolution_id 0 --compression_id 1 --test_resolution_id 1 --test_compression_id 2
# python test.py --model_id 1 --resolution_id 0 --compression_id 1 --test_resolution_id 0 --test_compression_id 2
# python test.py --model_id 1 --resolution_id 0 --compression_id 1 --test_resolution_id 1 --test_compression_id 2
# python test.py --model_id 2 --resolution_id 0 --compression_id 1 --test_resolution_id 0 --test_compression_id 2
# python test.py --model_id 2 --resolution_id 0 --compression_id 1 --test_resolution_id 1 --test_compression_id 2
# python test.py --model_id 3 --resolution_id 0 --compression_id 1 --test_resolution_id 0 --test_compression_id 2
# python test.py --model_id 3 --resolution_id 0 --compression_id 1 --test_resolution_id 1 --test_compression_id 2
# python test.py --model_id 4 --resolution_id 0 --compression_id 1 --test_resolution_id 0 --test_compression_id 2
# python test.py --model_id 4 --resolution_id 0 --compression_id 1 --test_resolution_id 1 --test_compression_id 2
# python test.py --model_id 5 --resolution_id 0 --compression_id 1 --test_resolution_id 0 --test_compression_id 2
# python test.py --model_id 5 --resolution_id 0 --compression_id 1 --test_resolution_id 1 --test_compression_id 2

# # # Train: HR; c23                     Test: LR and HR; c40;
# python test.py --model_id 0 --resolution_id 1 --compression_id 1 --test_resolution_id 0 --test_compression_id 2
# python test.py --model_id 0 --resolution_id 1 --compression_id 1 --test_resolution_id 1 --test_compression_id 2
# python test.py --model_id 1 --resolution_id 1 --compression_id 1 --test_resolution_id 0 --test_compression_id 2
# python test.py --model_id 1 --resolution_id 1 --compression_id 1 --test_resolution_id 1 --test_compression_id 2
# python test.py --model_id 2 --resolution_id 1 --compression_id 1 --test_resolution_id 0 --test_compression_id 2
# python test.py --model_id 2 --resolution_id 1 --compression_id 1 --test_resolution_id 1 --test_compression_id 2
# python test.py --model_id 3 --resolution_id 1 --compression_id 1 --test_resolution_id 0 --test_compression_id 2
# python test.py --model_id 3 --resolution_id 1 --compression_id 1 --test_resolution_id 1 --test_compression_id 2
# python test.py --model_id 4 --resolution_id 1 --compression_id 1 --test_resolution_id 0 --test_compression_id 2
# python test.py --model_id 4 --resolution_id 1 --compression_id 1 --test_resolution_id 1 --test_compression_id 2
# python test.py --model_id 5 --resolution_id 1 --compression_id 1 --test_resolution_id 0 --test_compression_id 2
# python test.py --model_id 5 --resolution_id 1 --compression_id 1 --test_resolution_id 1 --test_compression_id 2

# ###################################################################

# ##### Train c40 #####

# # # Train: LR; c40                     Test: LR and HR; c0;
# python test.py --model_id 0 --resolution_id 0 --compression_id 2 --test_resolution_id 0 --test_compression_id 0
# python test.py --model_id 0 --resolution_id 0 --compression_id 2 --test_resolution_id 1 --test_compression_id 0
# python test.py --model_id 1 --resolution_id 0 --compression_id 2 --test_resolution_id 0 --test_compression_id 0
# python test.py --model_id 1 --resolution_id 0 --compression_id 2 --test_resolution_id 1 --test_compression_id 0
# python test.py --model_id 2 --resolution_id 0 --compression_id 2 --test_resolution_id 0 --test_compression_id 0
# python test.py --model_id 2 --resolution_id 0 --compression_id 2 --test_resolution_id 1 --test_compression_id 0
# python test.py --model_id 3 --resolution_id 0 --compression_id 2 --test_resolution_id 0 --test_compression_id 0
# python test.py --model_id 3 --resolution_id 0 --compression_id 2 --test_resolution_id 1 --test_compression_id 0
# python test.py --model_id 4 --resolution_id 0 --compression_id 2 --test_resolution_id 0 --test_compression_id 0
# python test.py --model_id 4 --resolution_id 0 --compression_id 2 --test_resolution_id 1 --test_compression_id 0
# python test.py --model_id 5 --resolution_id 0 --compression_id 2 --test_resolution_id 0 --test_compression_id 0
# python test.py --model_id 5 --resolution_id 0 --compression_id 2 --test_resolution_id 1 --test_compression_id 0

# # # Train: HR; c40                     Test: LR and HR; c0;
# python test.py --model_id 0 --resolution_id 1 --compression_id 2 --test_resolution_id 0 --test_compression_id 0
# python test.py --model_id 0 --resolution_id 1 --compression_id 2 --test_resolution_id 1 --test_compression_id 0
# python test.py --model_id 1 --resolution_id 1 --compression_id 2 --test_resolution_id 0 --test_compression_id 0
# python test.py --model_id 1 --resolution_id 1 --compression_id 2 --test_resolution_id 1 --test_compression_id 0
# python test.py --model_id 2 --resolution_id 1 --compression_id 2 --test_resolution_id 0 --test_compression_id 0
# python test.py --model_id 2 --resolution_id 1 --compression_id 2 --test_resolution_id 1 --test_compression_id 0
# python test.py --model_id 3 --resolution_id 1 --compression_id 2 --test_resolution_id 0 --test_compression_id 0
# python test.py --model_id 3 --resolution_id 1 --compression_id 2 --test_resolution_id 1 --test_compression_id 0
# python test.py --model_id 4 --resolution_id 1 --compression_id 2 --test_resolution_id 0 --test_compression_id 0
# python test.py --model_id 4 --resolution_id 1 --compression_id 2 --test_resolution_id 1 --test_compression_id 0
# python test.py --model_id 5 --resolution_id 1 --compression_id 2 --test_resolution_id 0 --test_compression_id 0
# python test.py --model_id 5 --resolution_id 1 --compression_id 2 --test_resolution_id 1 --test_compression_id 0

# # # Train: LR; c40                     Test: LR and HR; c23;
# python test.py --model_id 0 --resolution_id 0 --compression_id 2 --test_resolution_id 0 --test_compression_id 1
# python test.py --model_id 0 --resolution_id 0 --compression_id 2 --test_resolution_id 1 --test_compression_id 1
# python test.py --model_id 1 --resolution_id 0 --compression_id 2 --test_resolution_id 0 --test_compression_id 1
# python test.py --model_id 1 --resolution_id 0 --compression_id 2 --test_resolution_id 1 --test_compression_id 1
# python test.py --model_id 2 --resolution_id 0 --compression_id 2 --test_resolution_id 0 --test_compression_id 1
# python test.py --model_id 2 --resolution_id 0 --compression_id 2 --test_resolution_id 1 --test_compression_id 1
# python test.py --model_id 3 --resolution_id 0 --compression_id 2 --test_resolution_id 0 --test_compression_id 1
# python test.py --model_id 3 --resolution_id 0 --compression_id 2 --test_resolution_id 1 --test_compression_id 1
# python test.py --model_id 4 --resolution_id 0 --compression_id 2 --test_resolution_id 0 --test_compression_id 1
# python test.py --model_id 4 --resolution_id 0 --compression_id 2 --test_resolution_id 1 --test_compression_id 1
# python test.py --model_id 5 --resolution_id 0 --compression_id 2 --test_resolution_id 0 --test_compression_id 1
# python test.py --model_id 5 --resolution_id 0 --compression_id 2 --test_resolution_id 1 --test_compression_id 1

# # # Train: HR; c40                     Test: LR and HR; c23;
# python test.py --model_id 0 --resolution_id 1 --compression_id 2 --test_resolution_id 0 --test_compression_id 1
# python test.py --model_id 0 --resolution_id 1 --compression_id 2 --test_resolution_id 1 --test_compression_id 1
# python test.py --model_id 1 --resolution_id 1 --compression_id 2 --test_resolution_id 0 --test_compression_id 1
# python test.py --model_id 1 --resolution_id 1 --compression_id 2 --test_resolution_id 1 --test_compression_id 1
# python test.py --model_id 2 --resolution_id 1 --compression_id 2 --test_resolution_id 0 --test_compression_id 1
# python test.py --model_id 2 --resolution_id 1 --compression_id 2 --test_resolution_id 1 --test_compression_id 1
# python test.py --model_id 3 --resolution_id 1 --compression_id 2 --test_resolution_id 0 --test_compression_id 1
# python test.py --model_id 3 --resolution_id 1 --compression_id 2 --test_resolution_id 1 --test_compression_id 1
# python test.py --model_id 4 --resolution_id 1 --compression_id 2 --test_resolution_id 0 --test_compression_id 1
# python test.py --model_id 4 --resolution_id 1 --compression_id 2 --test_resolution_id 1 --test_compression_id 1
# python test.py --model_id 5 --resolution_id 1 --compression_id 2 --test_resolution_id 0 --test_compression_id 1
# python test.py --model_id 5 --resolution_id 1 --compression_id 2 --test_resolution_id 1 --test_compression_id 1


# # # Train: LR; c40                     Test: LR and HR; c40;
# python test.py --model_id 0 --resolution_id 0 --compression_id 2 --test_resolution_id 0 --test_compression_id 2
# python test.py --model_id 0 --resolution_id 0 --compression_id 2 --test_resolution_id 1 --test_compression_id 2
# python test.py --model_id 1 --resolution_id 0 --compression_id 2 --test_resolution_id 0 --test_compression_id 2
# python test.py --model_id 1 --resolution_id 0 --compression_id 2 --test_resolution_id 1 --test_compression_id 2
# python test.py --model_id 2 --resolution_id 0 --compression_id 2 --test_resolution_id 0 --test_compression_id 2
# python test.py --model_id 2 --resolution_id 0 --compression_id 2 --test_resolution_id 1 --test_compression_id 2
# python test.py --model_id 3 --resolution_id 0 --compression_id 2 --test_resolution_id 0 --test_compression_id 2
# python test.py --model_id 3 --resolution_id 0 --compression_id 2 --test_resolution_id 1 --test_compression_id 2
# python test.py --model_id 4 --resolution_id 0 --compression_id 2 --test_resolution_id 0 --test_compression_id 2
# python test.py --model_id 4 --resolution_id 0 --compression_id 2 --test_resolution_id 1 --test_compression_id 2
# python test.py --model_id 5 --resolution_id 0 --compression_id 2 --test_resolution_id 0 --test_compression_id 2
# python test.py --model_id 5 --resolution_id 0 --compression_id 2 --test_resolution_id 1 --test_compression_id 2

# # # Train: HR; c40                     Test: LR and HR; c40;
# python test.py --model_id 0 --resolution_id 1 --compression_id 2 --test_resolution_id 0 --test_compression_id 2
# python test.py --model_id 0 --resolution_id 1 --compression_id 2 --test_resolution_id 1 --test_compression_id 2
# python test.py --model_id 1 --resolution_id 1 --compression_id 2 --test_resolution_id 0 --test_compression_id 2
# python test.py --model_id 1 --resolution_id 1 --compression_id 2 --test_resolution_id 1 --test_compression_id 2
# python test.py --model_id 2 --resolution_id 1 --compression_id 2 --test_resolution_id 0 --test_compression_id 2
# python test.py --model_id 2 --resolution_id 1 --compression_id 2 --test_resolution_id 1 --test_compression_id 2
# python test.py --model_id 3 --resolution_id 1 --compression_id 2 --test_resolution_id 0 --test_compression_id 2
# python test.py --model_id 3 --resolution_id 1 --compression_id 2 --test_resolution_id 1 --test_compression_id 2
# python test.py --model_id 4 --resolution_id 1 --compression_id 2 --test_resolution_id 0 --test_compression_id 2
# python test.py --model_id 4 --resolution_id 1 --compression_id 2 --test_resolution_id 1 --test_compression_id 2
# python test.py --model_id 5 --resolution_id 1 --compression_id 2 --test_resolution_id 0 --test_compression_id 2 
# python test.py --model_id 5 --resolution_id 1 --compression_id 2 --test_resolution_id 1 --test_compression_id 2

# ###################################################################

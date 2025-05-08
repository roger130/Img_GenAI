class Config:
    SEED = 42
    #random noise embedding dimension
    LATENT_DIM = 100  
    #img resolution
    IMAGE_SIZE = 64   
    CHANNELS = 1     
    NGF = 64          
    NDF = 64          
    
    #training parameters
    BATCH_SIZE = 64 
    LR = 0.0002       
    #adam optimizer parameter    
    BETA1 = 0.5   
    NUM_EPOCHS = 20
    
    RESULTS_DIR = './results'              
    MODELS_DIR = './models'               
    INDIVIDUAL_IMAGES_DIR = './individual_images'  
    EVAL_DIR = './evaluation'              
     
    FID_BATCH_SIZE = 64       
    NUM_FID_IMAGES = 1000     
    FID_EVERY = 5             
    
    NUM_INDIVIDUAL_IMAGES = 10  
    PRINT_EVERY = 30          
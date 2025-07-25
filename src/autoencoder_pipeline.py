import os
import pandas as pd
import numpy as np
from enum import Enum
from tqdm import tqdm
from pathlib import Path
from experiments.autoencoder.features_aggregator import FeaturesAggregator
from experiments.autoencoder.constants import QUERY_COLUMN, EMBEDDINGS_DTYPE, EventTypes
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import argparse

"""
I experimented with this based on loose tutorials and adapting them to our calculators feature interface. The config class up top is very handy for having all of the config in one place, due to our development happening in parralel the different methods are a bit diffuse in terms of implementation unity.
"""

class Config:

    DATA_DIR = "../data/original/"
    CLIENT_IDS_PATH = os.path.join(DATA_DIR, "input/relevant_clients.npy") 
    OUTPUT_DIR = "./output_deep_100ep"

    SAVE_RAW_FEATURES = False # Whether to save the raw features after engineering to the below folder or not
    SAVE_RAW_FEATURES_DIR = "./raw_features_2"
    EMBEDDINGS_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "embeddings.npy")
    IS_TEST_RUN = False #Use to run a test run with way fewer samples (relevant client ids), controlled by the below param
    TEST_SAMPLE_SIZE = 2000
   
    STATS_NUM_DAYS = [1, 2, 7, 30, 90]
    STATS_TOP_N = 10

    LOAD_FROM_EXISTING_EMBEDDINGS = False  #Skip feature engineering and load existing features in .npy form from the folder below. Note that it needs to have both "embeddings.npy" and "client_ids.npy" files.
    INPUT_FEATURES_DIR = "./raw_features_2"
    
    # Autoencoder params
    USE_DEEPER_AUTOENCODER = True #Simple switch between the normal autoencoder model and the one with one more layer
    USE_LR_SCHEDULING = True #Whether to use learning rate scheduling during training
    EMBEDDING_DIM = 180 #Output dimensions of the autoencoded features
    AE_EPOCHS = 120 
    AE_BATCH_SIZE = 256
    AE_LEARNING_RATE = 1e-3




EVENT_TYPE_TO_FILENAME = {
    EventTypes.PRODUCT_BUY: "product_buy.parquet",
    EventTypes.ADD_TO_CART: "add_to_cart.parquet",
    EventTypes.SEARCH_QUERY: "search_query.parquet",
    EventTypes.PAGE_VISIT: "page_visit.parquet", #ran without for debugging
}


EVENT_TYPE_TO_COLUMNS = {
    EventTypes.PRODUCT_BUY: ['sku', 'category'],
    EventTypes.ADD_TO_CART: ['sku', 'category'],
    EventTypes.SEARCH_QUERY: ['query'],
    EventTypes.PAGE_VISIT: ['url'],
}



class Autoencoder(nn.Module):
    """mishmash of different sources, but mostly this https://www.datacamp.com/tutorial/introduction-to-autoencoders, building on the basic one,   and adding dropout to maybe get some of the benefits of a denoising autoencoder????"""
    def __init__(self, input_dim, embedding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), 
            nn.ReLU(),
            nn.Dropout(0.2), # regularization, can maybe act as a sort of denoising,
            nn.Linear(256, embedding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid() # assumes input was scaled to [0,1]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def get_embeddings(self, x):
        
        return self.encoder(x) # uses the encoder to get the final embeddings here


class Autoencoder_deeper(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(Autoencoder_deeper, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),    #todo: test with wider input   
            nn.ReLU(),
            nn.Linear(256, 256),             
            nn.ReLU(),
            nn.Dropout(0.2),                 
            nn.Linear(256, embedding_dim)   
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),             
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def get_embeddings(self, x):
        
        return self.encoder(x) # uses the encoder to get the final embeddings here


class Autoencoder_wider_deeper(nn.Module): #add dropout twice and batch normalization, common recommendations to make a model generalize better
    def __init__(self, input_dim, embedding_dim):
        super(Autoencoder_wider_deeper, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),   #wider input layer
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            
            nn.Linear(512, 256),             
            nn.ReLU(),
            nn.Linear(256, 256),             
            nn.ReLU(),
            nn.Dropout(0.2),                 
            nn.Linear(256, embedding_dim)   
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            
            nn.Linear(512, 256),             
            nn.ReLU(),
            nn.Linear(256, 512),             
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def get_embeddings(self, x):
        
        return self.encoder(x) # uses the encoder to get the final embeddings here

    

def main(params):

    if(params):
        data_dir_arg = params.data_dir
    
        embeddings_dir_arg = params.embeddings_dir
    
        if(data_dir_arg):
            Config.DATA_DIR = data_dir_arg
        print(f"configured data dir to: {Config.DATA_DIR}")
    
        if(embeddings_dir_arg): 
            Config.OUTPUT_DIR = embeddings_dir_arg
        print(f"configured embeddings dir to: {Config.OUTPUT_DIR}")

   
    print("\n Loading and pre-processing data...")
    print("USING INPUT DATA FOLDER: " + Config.DATA_DIR)
    print("USING CLIENT IDS PATH: " + Config.CLIENT_IDS_PATH)
    output_dir = Path(Config.OUTPUT_DIR)
    # Ensure output directory exists
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    if not (Config.LOAD_FROM_EXISTING_EMBEDDINGS):
    
        if not os.path.exists(Config.CLIENT_IDS_PATH):
            raise FileNotFoundError(f"Crucial file not found: {Config.CLIENT_IDS_PATH}. This file is required for final ordering.")
        relevant_client_ids = np.load(Config.CLIENT_IDS_PATH, allow_pickle=True)
    
        
    
        if Config.IS_TEST_RUN:
            print("TEST RUN WITH FEWER USERS") 
            if len(relevant_client_ids) > Config.TEST_SAMPLE_SIZE:
                relevant_client_ids = np.random.choice(relevant_client_ids, Config.TEST_SAMPLE_SIZE, replace=False)
            else:
                print("Warning: Sample size larger than all clients")
                
        else:
            pass
        
        product_properties_df = pd.read_parquet(os.path.join(Config.DATA_DIR, "product_properties.parquet"))
        
        all_event_dfs = {}
        for event_type, filename in EVENT_TYPE_TO_FILENAME.items():
            path = os.path.join(Config.DATA_DIR, filename)
            if os.path.exists(path):
                print(f"  - Loading {filename}...")
                df = pd.read_parquet(path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                all_event_dfs[event_type] = df
            else:
                print(f"  - Warning: {filename} not found, skipping.")
    
        #  Merge product properties
        props_to_merge = product_properties_df[['sku', 'price', 'category', 'name']].copy()
        if EventTypes.PRODUCT_BUY in all_event_dfs:
            all_event_dfs[EventTypes.PRODUCT_BUY] = all_event_dfs[EventTypes.PRODUCT_BUY].merge(props_to_merge, on='sku', how='left')
        if EventTypes.ADD_TO_CART in all_event_dfs:
            all_event_dfs[EventTypes.ADD_TO_CART] = all_event_dfs[EventTypes.ADD_TO_CART].merge(props_to_merge, on='sku', how='left')
        print("  - Merged product properties with buy and cart events.")
    
        
        print("\n Aggregating features")
        feature_agg = FeaturesAggregator( #instantiate featuresagg with params from above
            num_days=Config.STATS_NUM_DAYS,
            top_n=Config.STATS_TOP_N,
            relevant_client_ids=relevant_client_ids
        )
    
        buy_events_df = all_event_dfs.get(EventTypes.PRODUCT_BUY)
    
        for event_type, df in all_event_dfs.items(): #loop over the feature types and generate features
            print(f"  - generating features for event type: {event_type.value}...")
            feature_agg.generate_features(
                event_type=event_type,
                client_id_column="client_id",
                df=df,
                columns=EVENT_TYPE_TO_COLUMNS.get(event_type, []),
                product_properties=product_properties_df, # Pass for reference if any calc needs it
                buy_events=buy_events_df # For CartAbandonmentCalculator
            )
    
        # Postprocessing steps like aligning, scaling
        print("\n Postprocessing steps like aligning, scaling")
        client_ids_from_agg, all_user_features_raw = feature_agg.merge_features()
        print(f"  -Feature shape: {all_user_features_raw.shape}")
        
        raw_features_df = pd.DataFrame(all_user_features_raw, index=client_ids_from_agg)
        #IMPORTANT!!!!!! Align rows to the official client_ids.npy order.
        ordered_features_df = raw_features_df.reindex(relevant_client_ids)
        
        # Fill any nans, from reindexing or calculations
        all_user_features_filled = ordered_features_df.fillna(0).values

        if(Config.SAVE_RAW_FEATURES):
            try:
    
                raw_feat_dir = Path(Config.SAVE_RAW_FEATURES_DIR)
                raw_feat_dir.mkdir(parents=True, exist_ok=True) 
                embeddings_path = raw_feat_dir / "embeddings.npy"
                client_ids_path = raw_feat_dir / "client_ids.npy"
                client_ids_to_save = relevant_client_ids #should this be client_ids_from_agg? 
        
                
                np.save(embeddings_path, all_user_features_filled)
                np.save(client_ids_path, client_ids_to_save)
            except Exception as e:
                print("ERROR during attempt to save raw features") 
                print(e)

    
    else: 
        
         
        print("LOADING EXISTING RAW FEATURE EMBEDDINGS")
        
        input_embeddings_path = Path(Config.INPUT_FEATURES_DIR) / "embeddings.npy"
        input_clients_path = Path(Config.INPUT_FEATURES_DIR) / "client_ids.npy"
        
        if not input_embeddings_path.exists() or not input_clients_path.exists():
            raise FileNotFoundError(f"Input files not found in {Config.INPUT_FEATURES_DIR}. needs to contain embeddings.npy and client_ids.npy.")
    

        all_user_features_raw = np.load(input_embeddings_path)
        client_ids_raw = np.load(input_clients_path)
        # Fill any nans, from reindexing or calculations, unsure if necessary here?
        #all_user_features_filled = ordered_features_df.fillna(0).values
        
        relevant_client_ids = np.load(input_clients_path)
        client_ids_to_save = relevant_client_ids
        print(f"  - Loaded raw features of shape: {all_user_features_raw.shape} FROM {input_embeddings_path}")
        assert len(all_user_features_raw) == len(client_ids_to_save), "Mismatch between embeddings and client_ids length!"



        raw_features_df = pd.DataFrame(all_user_features_raw, index=client_ids_raw)
        #IMPORTANT!!!!!! Align rows to the official client_ids.npy order.
        ordered_features_df = raw_features_df.reindex(relevant_client_ids)
        
        # Fill any nans, from reindexing or calculations
        all_user_features_filled = ordered_features_df.fillna(0).values
        #all_user_features_filled = all_user_features_raw 
        
    
    # Scale features to be between 0 and 1 for the autoencoder
    scaler = MinMaxScaler()
    all_user_features_scaled = scaler.fit_transform(all_user_features_filled)
    print(f"  - Final scaled feature matrix shape: {all_user_features_scaled.shape}")

    # ~~~~~~~ TRAINING STEP ~~~~
    print("\n Training the Autoencoder...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  - device: {device}")

    features_tensor = torch.tensor(all_user_features_scaled, dtype=torch.float32)
    dataset = TensorDataset(features_tensor)
    dataloader = DataLoader(dataset, batch_size=Config.AE_BATCH_SIZE, shuffle=True)
    
    input_dim_ae = all_user_features_scaled.shape[1]

    if(Config.USE_DEEPER_AUTOENCODER):
        autoencoder = Autoencoder_deeper(input_dim_ae, Config.EMBEDDING_DIM).to(device)
        print(f"Using the deeper autoencoder model, embedding dim {Config.EMBEDDING_DIM}")
    else:  
        autoencoder = Autoencoder(input_dim_ae, Config.EMBEDDING_DIM).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=Config.AE_LEARNING_RATE)
    # try scheduling the LR: https://stackoverflow.com/questions/63108131/pytorch-schedule-learning-rate 
    if(Config.USE_LR_SCHEDULING):
        #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, threshold=Config.AE_LEARNING_RATE, verbose=True)

        scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=Config.AE_LEARNING_RATE, 
        steps_per_epoch=len(dataloader), 
        epochs=Config.AE_EPOCHS
        )
        print("Using LR Scheduling")
    
    autoencoder.train()
    for epoch in range(Config.AE_EPOCHS):
        epoch_loss = 0.0
        for data in dataloader:
            batch_features = data[0].to(device)
            optimizer.zero_grad()
            outputs = autoencoder(batch_features)
            loss = criterion(outputs, batch_features)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_features.size(0)
            if(Config.USE_LR_SCHEDULING):
                scheduler.step()
        epoch_loss /= len(dataloader.dataset)
        print(f"  - Epoch [{epoch+1}/{Config.AE_EPOCHS}], Loss: {epoch_loss:.6f}")
        if(Config.USE_LR_SCHEDULING):
            print(f"learning rate: {str(scheduler.get_last_lr())}")
            pass
            #print("scheduler step")
            #scheduler.step(epoch_loss)

    # Generate fiinal embeddings 
    print("\n Generating final embeddings")
    autoencoder.eval()
    with torch.no_grad():
        all_features_on_device = features_tensor.to(device)
        final_embeddings_tensor = autoencoder.get_embeddings(all_features_on_device)
        final_embeddings_numpy = final_embeddings_tensor.cpu().numpy()

    #Last step: Save embeddings
    print("\n Saving final embeddings (after converting to float16 to appease the validator script)")


    final_embeddings_numpy = final_embeddings_numpy.astype(np.float16) #validator said it wants fp16
 
    output_dir = Path(Config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True) 
    embeddings_path = output_dir / "embeddings.npy"
    client_ids_path = output_dir / "client_ids.npy"
    

    client_ids_to_save = relevant_client_ids

    
    np.save(embeddings_path, final_embeddings_numpy)
    np.save(client_ids_path, client_ids_to_save)


    if(1 == 2):
        #EXPERIMENT: Concatenating them side by side with the original features, like gabriel's idea
        scaler_raw = MinMaxScaler()
        scaled_raw_features = scaler_raw.fit_transform(all_user_features_filled)
        
        scaler_ae = MinMaxScaler()
        scaled_ae_embeddings = scaler_ae.fit_transform(final_embeddings_numpy)
        
        # Concatenate them side-by-side
        hybrid_embeddings = np.concatenate([scaled_raw_features, scaled_ae_embeddings], axis=1)
        
        print(f"Final hybrid embedding shape: {hybrid_embeddings.shape}") # Should be (N, 1430)
        
        # Convert to float16 and save this new hybrid embedding
        hybrid_embeddings_f16 = hybrid_embeddings.astype(np.float16)
        np.save(output_dir / "hybrid_embeddings.npy", hybrid_embeddings_f16)


    
    print("SUCCESS!")
    print(f"Embeddings of shape {final_embeddings_numpy.shape} saved to:")
    print(f"{embeddings_path}")
    print(f"Client IDs of shape {client_ids_to_save.shape} saved to:")
    print(f"{client_ids_path}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        required=False,
        help="Directory with input and target data – produced by data_utils.split_data",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        required=False,
        help="Directory where to store generated embeddings",
    )
    params = parser.parse_args()
    main(params=params)
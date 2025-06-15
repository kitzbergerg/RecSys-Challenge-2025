import os
import pandas as pd
import numpy as np
from enum import Enum
from tqdm import tqdm
from pathlib import Path
from custom_pipline.features_aggregator import FeaturesAggregator
from custom_pipline.constants import QUERY_COLUMN, EMBEDDINGS_DTYPE, EventTypes
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler


"""
I experimented with this based on loose tutorials and adapting them to our calculators feature interface. The config class up top is very handy for having all of the config in one place, due to our development happening in parralel the different methods are a bit diffuse in terms of implementation unity.
"""

class Config:

    DATA_DIR = "/home/jovyan/shared/194.035-2025S/data/group_project/data_new/"
    CLIENT_IDS_PATH = os.path.join(DATA_DIR, "input/relevant_clients.npy") 
    OUTPUT_DIR = "./output_new"
    EMBEDDINGS_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "embeddings.npy")
    IS_TEST_RUN = False
    TEST_SAMPLE_SIZE = 2000
   
    STATS_NUM_DAYS = [7, 30, 90]
    STATS_TOP_N = 10

    
    
    # Autoencoder params
    USE_DEEPER_AUTOENCODER = True
    USE_LR_SCHEDULING = True
    EMBEDDING_DIM = 128 
    AE_EPOCHS = 50
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
    EventTypes.SEARCH_QUERY: [QUERY_COLUMN], # QUERY_COLUMN is likely 'query'
    EventTypes.PAGE_VISIT: ['url'],
}



class Autoencoder(nn.Module):
    """mishmash of different sources, but mostly this https://www.datacamp.com/tutorial/introduction-to-autoencoders"""
    def __init__(self, input_dim, embedding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), # Intermediate layer
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, embedding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid() # Assumes input was scaled to [0,1] by MinMaxScaler
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
            nn.Linear(input_dim, 512),       #wider than before
            nn.ReLU(),
            nn.Linear(512, 256),             
            nn.ReLU(),
            nn.Dropout(0.2),                 
            nn.Linear(256, embedding_dim)    #bottleneck
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 256),
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

    

def main():

    
   
    print("\n Loading and pre-processing data...")
    print("USING INPUT DATA FOLDER: " + Config.DATA_DIR)
    print("USING CLIENT IDS PATH: " + Config.CLIENT_IDS_PATH)
    output_dir = Path(Config.OUTPUT_DIR)
    # Ensure output directory exists
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
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
        print("Using the deeper autoencoder model, one extra layer babyyy")
    else:  
        autoencoder = Autoencoder(input_dim_ae, Config.EMBEDDING_DIM).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=Config.AE_LEARNING_RATE)
    # try scheduling the LR: https://stackoverflow.com/questions/63108131/pytorch-schedule-learning-rate 
    if(Config.USE_LR_SCHEDULING):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, verbose=True)
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
        
        epoch_loss /= len(dataloader.dataset)
        print(f"  - Epoch [{epoch+1}/{Config.AE_EPOCHS}], Loss: {epoch_loss:.6f}")
        if(Config.USE_LR_SCHEDULING):
            scheduler.step(epoch_loss)

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

    print("SUCCESS!")
    print(f"Embeddings of shape {final_embeddings_numpy.shape} saved to:")
    print(f"{embeddings_path}")
    print(f"Client IDs of shape {client_ids_to_save.shape} saved to:")
    print(f"{client_ids_path}")

if __name__ == "__main__":
    main()
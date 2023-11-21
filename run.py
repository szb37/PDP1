import src.format_data as data
import src.folders as folders
import src.config as config

# Create master DF for all downstream analysis from raw data files
df = data.Controllers.get_master_df()

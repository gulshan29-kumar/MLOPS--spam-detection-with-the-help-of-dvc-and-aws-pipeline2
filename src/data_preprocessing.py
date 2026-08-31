import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk
import yaml
nltk.download('stopwords')
nltk.download('punkt')

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('Parameters file not found: %s', params_path)
        raise FileNotFoundError(f"Configuration file {params_path} could not be located.")
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise ValueError(f"Invalid YAML config file format: {e}")
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise e

# Setting up logger
logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text: str, lowercase: bool = True, stopwords_lang: str = 'english') -> str:
    """
    Transforms the input text by converting it to lowercase, tokenizing, removing stopwords 
    and punctuation, and stemming using PorterStemmer.
    
    Args:
        text (str): Raw message clean text.
        lowercase (bool): Whether to convert text to lower case. Defaults to True.
        stopwords_lang (str): Stopwords language. Defaults to 'english'.
        
    Returns:
        str: Fully processed and stemmed text.
    """
    ps = PorterStemmer()
    # Convert to lowercase
    if lowercase:
        text = text.lower()
    # Tokenize the text
    text = nltk.word_tokenize(text)
    # Remove non-alphanumeric tokens
    text = [word for word in text if word.isalnum()]
    # Remove stopwords and punctuation
    text = [word for word in text if word not in stopwords.words(stopwords_lang) and word not in string.punctuation]
    # Stem the words
    text = [ps.stem(word) for word in text]
    # Join the tokens back into a single string
    return " ".join(text)

def preprocess_df(df: pd.DataFrame, text_column: str = 'text', target_column: str = 'target', lowercase: bool = True, stopwords_lang: str = 'english') -> pd.DataFrame:
    """
    Preprocesses the DataFrame by encoding the target column, removing duplicates, 
    and transforming the text column.
    
    Args:
        df (pd.DataFrame): Input training or test split.
        text_column (str): Label for the input text feature. Defaults to 'text'.
        target_column (str): Target classification label. Defaults to 'target'.
        lowercase (bool): Convert message text to lowercase. Defaults to True.
        stopwords_lang (str): Stopwords language filter. Defaults to 'english'.
        
    Returns:
        pd.DataFrame: Transformed pandas DataFrame.
    """
    try:
        logger.debug('Starting preprocessing for DataFrame')
        # Encode the target column
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug('Target column encoded')

        # Remove duplicate rows
        df = df.drop_duplicates(keep='first')
        logger.debug('Duplicates removed')
        
        # Apply text transformation to the specified text column
        df.loc[:, text_column] = df[text_column].apply(lambda x: transform_text(x, lowercase, stopwords_lang))
        logger.debug('Text column transformed')
        return df
    
    except KeyError as e:
        logger.error('Column not found: %s', e)
        raise KeyError(f"Expected column could not be found: {e}")
    except Exception as e:
        logger.error('Error during text normalization: %s', e)
        raise

def main(text_column: str = 'text', target_column: str = 'target') -> None:
    """
    Main preprocessing routine. Load files from raw path, normalize text fields,
    and save outputs into data/interim.
    
    Args:
        text_column (str): Label of feature text column. Defaults to 'text'.
        target_column (str): Label of target column. Defaults to 'target'.
    """
    try:
        # Load parameters
        params = load_params('params.yaml').get('data_preprocessing', {})
        lowercase = params.get('lowercase', True)
        stopwords_lang = params.get('stopwords_lang', 'english')

        # Fetch the data from data/raw
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('Data loaded properly')

        # Transform the data
        train_processed_data = preprocess_df(train_data, text_column, target_column, lowercase, stopwords_lang)
        test_processed_data = preprocess_df(test_data, text_column, target_column, lowercase, stopwords_lang)

        # Store the data inside data/processed
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)
        
        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        
        logger.debug('Processed data saved to %s', data_path)
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s', e)
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()

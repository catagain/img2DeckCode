import json
import requests
import os

def generate_id_to_data_map():
    """
    Fetches card data from YGOPRODeck API and generates a mapping file
    linking every card ID (including alternate arts) to its card name.
    """
    url = "https://db.ygoprodeck.com/api/v7/cardinfo.php"
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    project_root = os.path.dirname(script_dir)
    data_folder = os.path.join(project_root, 'data')
    output_file = os.path.join(data_folder, 'id_to_card_data.json')
    
    if not os.path.exists(data_folder):
        print(f"Creating missing directory: {data_folder}")
        os.makedirs(data_folder, exist_ok=True)

    try:
        print(f"Fetching card data from {url}...")
        response = requests.get(url, timeout=10)
        
        response.raise_for_status()
        
        raw_data = response.json()
        cards = raw_data.get('data', [])
        
        if not cards:
            print("The API returned an empty data list.")
            return

        id_to_data = {}
        for card in cards:
            card_name = card.get('name')
            # Remove double quotes because they may cause errors in database queries
            clean_name = card_name.replace('"', '')

            card_type = card.get('type', 'Unknown')

            card_info = {
                "name": clean_name,
                "type": card_type
            }
            
            main_id = str(card.get('id'))
            id_to_data[main_id] = card_info
            print(f'Add card:\"{clean_name}\" with type:\"{card_type}\", id:{main_id}')

        # Write the resulting dictionary to a JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(id_to_data, f, ensure_ascii=False, indent=4)
        
        print(f"Success! Mapping generated with {len(id_to_data)} entries.")

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except requests.exceptions.ConnectionError:
        print("Connection error: Please check your internet connectivity.")
    except requests.exceptions.Timeout:
        print("Timeout error: The server took too long to respond.")
    except json.JSONDecodeError:
        print("JSON error: Failed to parse the API response.")
    except IOError as e:
        print(f"File error: Failed to write to {output_file}. Original error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Execute
if __name__ == "__main__":
    generate_id_to_data_map()
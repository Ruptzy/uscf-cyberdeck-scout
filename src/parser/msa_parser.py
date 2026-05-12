import re
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)

def parse_player_profile(html, source_url, source_file, target_player_id):
    """Extracts basic metadata from the Main Profile page."""
    soup = BeautifulSoup(html, 'html.parser')
    
    name, state, reg, quick, blitz = "Unknown", "Unknown", "Unknown", "Unknown", "Unknown"
    
    return {
        "uscf_id": target_player_id,
        "name": name,
        "state": state,
        "rating_regular": reg,
        "rating_quick": quick,
        "rating_blitz": blitz,
        "player_profile_url": source_url,
        "source_url": source_url,
        "source_html_file": source_file,
        "parse_status": "Success" 
    }

def parse_tournament_history(html, source_url, source_file):
    """Extracts a list of recent events from the Tournament History page."""
    soup = BeautifulSoup(html, 'html.parser')
    events = []
    seen_events = set()
    
    for a in soup.find_all('a', href=re.compile(r'XtblMain\.php\?')):
        href = a['href']
        match = re.search(r'XtblMain\.php\?([0-9\.]+)', href)
        if not match:
            continue
            
        event_id = match.group(1)
        if event_id in seen_events:
            continue
        seen_events.add(event_id)
        
        event_name = a.text.strip()
        
        row = a.find_parent('tr')
        end_date = "Unknown"
        if row:
            tds = row.find_all('td')
            # The date is usually in the first column like: "2023-01-01\n202301011234"
            for td in tds:
                date_match = re.search(r'\d{4}-\d{2}-\d{2}', td.text)
                if date_match:
                    end_date = date_match.group()
                    break
                
        events.append({
            "event_id": event_id,
            "event_name": event_name,
            "end_date": end_date,
            "raw_time_control_text": "Unknown", 
            "normalized_time_control": "Unknown",
            "source_url": source_url,
            "source_html_file": source_file,
            "parse_status": "Success"
        })
        
    return events

def parse_crosstable(html, source_url, source_file, target_player_id, event_id):
    """Parses USCF crosstables by reading the ASCII text inside the <pre> tag."""
    soup = BeautifulSoup(html, 'html.parser')
    games = []
    
    tc_text, tc_norm = "Unknown", "Unknown"
    tc_match = re.search(r'Time Control:?\s*([^\n<]+)', html, re.IGNORECASE)
    if tc_match:
        tc_text = tc_match.group(1).strip()
        tc_norm = "Blitz" if "d3" in tc_text or "G/3" in tc_text or "G/5" in tc_text else \
                  "Quick" if "G/1" in tc_text or "G/2" in tc_text else \
                  "Regular" if "G/6" in tc_text or "G/9" in tc_text else "Mixed"
                  
    pre_tag = soup.find('pre')
    if not pre_tag:
        return tc_text, tc_norm, games
        
    lines = pre_tag.text.split('\n')
    
    player_map = {}
    current_pair = None
    current_results = []
    
    for line in lines:
        if '|' not in line:
            continue
            
        parts = [p.strip() for p in line.split('|')]
        if len(parts) < 3:
            continue
            
        if parts[0].isdigit():
            current_pair = parts[0]
            current_results = parts[3:-1]
            
        elif current_pair and '/' in parts[1]:
            id_rating_str = parts[1]
            id_match = re.search(r'(\d{8})', id_rating_str)
            # Anchor rating on "R:" so we never confuse the 8-digit USCF ID
            # for a rating. Capture optional 'P<n>' provisional suffix so
            # downstream cleaning can strip it; treat 'Unrated' as missing.
            rating_match = re.search(r'R:\s*(\d{3,4}P?\d*)', id_rating_str)

            uscf_id = id_match.group(1) if id_match else "Unknown"
            pre_rating = rating_match.group(1) if rating_match else "Unknown"
            
            player_map[current_pair] = {
                "id": uscf_id,
                "pre_rating": pre_rating,
                "results": current_results
            }
            current_pair = None
            
    target_pairing = None
    player_pre = "Unknown"
    
    for p_num, p_data in player_map.items():
        if p_data["id"] == str(target_player_id):
            target_pairing = p_num
            player_pre = p_data["pre_rating"]
            break
            
    if not target_pairing:
        return tc_text, tc_norm, games
        
    target_results = player_map[target_pairing]["results"]
    
    for idx, result_str in enumerate(target_results):
        if not result_str.strip():
            continue
            
        res_match = re.match(r'([WLD])\s+(\d+)', result_str.strip())
        if not res_match:
            continue
            
        result_char = res_match.group(1)
        opp_num = res_match.group(2)
        
        opp_id = f"OPP_{opp_num}"
        opp_pre = "Unknown"
        
        if opp_num in player_map:
            opp_id = player_map[opp_num]["id"]
            opp_pre = player_map[opp_num]["pre_rating"]
            
        color = "Unknown" 
        
        parse_status = "Success"
        if opp_id.startswith("OPP_") or opp_pre == "Unknown" or player_pre == "Unknown":
            parse_status = "Partial"
            
        games.append({
            "game_id": f"{event_id}_{target_player_id}_R{idx+1}",
            "event_id": event_id,
            "player_id": target_player_id,
            "opponent_id": opp_id,
            "section_identifier": "Main",
            "round_num": idx + 1,
            "color": color,
            "player_pre_rating": player_pre,
            "opponent_pre_rating": opp_pre,
            "result": result_char,
            "player_post_rating": "Unknown",
            "opponent_post_rating": "Unknown",
            "raw_result_text": result_str,
            "source_url": source_url,
            "source_html_file": source_file,
            "parse_status": parse_status
        })
        
    return tc_text, tc_norm, games

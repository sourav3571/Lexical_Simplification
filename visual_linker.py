# visual_linker.py

import os
import json
import urllib.parse
import requests
from typing import Dict, Any, List
from nltk.corpus import wordnet as wn

# ---------------------------------------------------------------------------
# VISUAL_CONFIG
# ---------------------------------------------------------------------------
VISUAL_CONFIG = {
    "show_emoji": True,
    "show_image_link": True,
    "show_definition": True,
    "show_pronunciation": True,
    "show_audio_link": True,
    "show_example": True,
    "output_format": "html",
    "image_source": "unsplash",
    "tooltip_on_hover": True,
    "image_width": 300,
    "image_height": 200,
    "max_related_words": 3,
    "show_complexity_score": True,
    "show_original_word": True,
    "highlight_changes": True
}

# ---------------------------------------------------------------------------
# EmojiMapper Class
# ---------------------------------------------------------------------------
class EmojiMapper:
    def __init__(self):
        self.emoji_dict = {
            "physician": "👨‍⚕️",
            "doctor": "👨‍⚕️",
            "surgeon": "🏥",
            "lawyer": "⚖️",
            "defendant": "👨‍💼",
            "teacher": "👨‍🏫",
            "scientist": "👨‍🔬",
            "engineer": "👨‍💻",
            
            # Category: Actions/Verbs
            "utilized": "🔧",
            "commenced": "🚀",
            "ameliorate": "⬆️",
            "demonstrated": "👁️",
            "purchased": "🛒",
            "consumed": "🍽️",
            "administered": "💊",
            "endeavored": "💪",
            "acquired": "🤝",
            "exhibited": "🎭",
            
            # Category: Conditions/States
            "exhausted": "😴",
            "fatigued": "😫",
            "delighted": "😊",
            "precarious": "⚠️",
            "devastating": "💔",
            "overwhelming": "🌊",
            "enduring": "⏳",
            "remarkable": "⭐",
            "exceptional": "🌟",
            "resilience": "💪",
            
            # Category: Academic
            "hypothesis": "🔬",
            "methodology": "📋",
            "phenomenon": "🌀",
            "correlation": "📊",
            "validated": "✅",
            "significant": "📌",
            "empirically": "🧪",
            "paradigm": "🔄",
            "implications": "💭",
            "comprehensive": "📚",
            
            # Category: Medical
            "medication": "💊",
            "cardiovascular": "❤️",
            "neurological": "🧠",
            "diagnosis": "🔍",
            "symptoms": "🤒",
            "surgical": "🔪",
            "pharmaceutical": "🏥",
            "immunological": "🛡️",
            "procedure": "📋",
            "patient": "🤒",
            "patient_person": "🤒",
            "treat": "💊",
            "treated": "💊",
            "treatment": "💊",
            "medicine": "💊",
            "sick": "🤒",
            "ill": "🤒",
            "pain": "😢",
            "hospital": "🏥",
            "clinic": "🏥",
            "nurse": "🧑‍⚕️",
            "care": "❤️",
            "cure": "🧪",
            "disease": "🦠",
            "virus": "🦠",
            "injury": "🩹",
            "wound": "🩹",
            "health": "💚",
            "healthy": "💪",
            
            # People/Relationships
            "person": "👤",
            "man": "👨",
            "woman": "👩",
            "child": "🧒",
            "boy": "👦",
            "girl": "👧",
            "baby": "👶",
            "friend": "🤝",
            "family": "👨‍👩‍👧‍👦",
            "parent": "👪",
            "father": "👨",
            "mother": "👩",
            "brother": "👦",
            "sister": "👧",
            
            # Common Actions/Verbs
            "help": "🤝",
            "helper": "🤝",
            "work": "💼",
            "worker": "💼",
            "job": "💼",
            "do": "⚡",
            "make": "🛠️",
            "create": "🎨",
            "build": "🔨",
            "think": "🧠",
            "thought": "🧠",
            "know": "🧠",
            "learn": "📚",
            "study": "📖",
            "teach": "🧑‍🏫",
            "teacher": "🧑‍🏫",
            "find": "🔍",
            "search": "🔍",
            "look": "👀",
            "see": "👁️",
            "show": "👁️",
            "give": "🤲",
            "take": "🤲",
            "buy": "🛒",
            "sell": "🏪",
            "pay": "💵",
            "cost": "💰",
            "spend": "💵",
            "eat": "😋",
            "drink": "🥛",
            "sleep": "😴",
            "walk": "🚶",
            "run": "🏃",
            "go": "🏃",
            "come": "👋",
            "leave": "🚪",
            "stay": "🏠",
            "live": "🏠",
            "speak": "🗣️",
            "talk": "💬",
            "say": "💬",
            "tell": "💬",
            "write": "✍️",
            "read": "📖",
            "understand": "💡",
            
            # Common Nouns
            "time": "⏰",
            "day": "☀️",
            "night": "🌙",
            "week": "📅",
            "month": "📅",
            "year": "📅",
            "money": "💵",
            "cash": "💵",
            "problem": "⚠️",
            "solution": "💡",
            "idea": "💡",
            "question": "🤔",
            "answer": "✅",
            "food": "🍕",
            "water": "💧",
            "home": "🏠",
            "house": "🏠",
            "school": "🏫",
            "office": "🏢",
            "car": "🚗",
            "vehicle": "🚗",
            "city": "🏙️",
            "town": "🏡",
            "country": "🌍",
            "world": "🌍",
            "place": "📍",
            "location": "📍",
            
            # Common Adjectives/States
            "good": "👍",
            "bad": "👎",
            "happy": "😊",
            "sad": "😔",
            "angry": "😠",
            "tired": "🥱",
            "exhausted": "😴",
            "easy": "✅",
            "hard": "💪",
            "difficult": "⚠️",
            "fast": "⚡",
            "slow": "🐢",
            "big": "🐘",
            "large": "🐘",
            "small": "🐭",
            "little": "🐭",
            "new": "✨",
            "old": "👴",
            "hot": "🔥",
            "cold": "❄️",
            "warm": "☀️",
            "cool": "😎",
            "beautiful": "✨",
            "pretty": "✨",
            "ugly": "👹",
            "clean": "🧼",
            "dirty": "🧹",
            
            # Category: Idioms
            "kick the bucket": "⚰️",
            "kicked the bucket": "⚰️",
            "under the weather": "🤒",
            "spill the beans": "🫘",
            "spilled the beans": "🫘",
            "hit the nail on the head": "🔨",
            "hit the nail on head": "🔨",
            "bite the bullet": "💪",
            "bit the bullet": "💪",
            "let the cat out of the bag": "🐱",
            "let cat out of bag": "🐱",
            "burn the midnight oil": "🌙",
            "burned the midnight oil": "🌙",
            "beat around the bush": "🌿",
            "break the ice": "🧊",
            "broke the ice": "🧊",
            "cost an arm and a leg": "💰",
            "cost arm and leg": "💰",
        }

        # Category: Abstract/Figurative vs Literal mappings
        self.figurative_mappings = {
            "nature": "💭",       # figurative
            "spirit": "💡",       # figurative
            "heart": "🎯",        # figurative/core
            "face": "🌍",         # figurative
            "root": "🔍",         # figurative
            "shadow": "😟",        # figurative
            "fabric": "🏛️",        # figurative
            "weight": "😔",        # figurative
            "depth": "📚",         # figurative
        }
        self.literal_mappings = {
            "nature": "🌿",
            "spirit": "👻",
            "heart": "❤️",
            "face": "😊",
            "root": "🌱",
            "shadow": "🌑",
            "fabric": "🧵",
            "weight": "⚖️",
            "depth": "🌊",
        }

        # Load label_to_emoji.json if exists for fallback (6000+ words)
        self.fallback_dict = {}
        # Try paths relative to this file
        here = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.join(here, "label_to_emoji.json"),
            os.path.join(os.path.dirname(here), "frontend", "src", "data", "label_to_emoji.json"),
            os.path.join(os.getcwd(), "frontend", "src", "data", "label_to_emoji.json"),
            "../frontend/src/data/label_to_emoji.json"
        ]
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        self.fallback_dict = json.load(f)
                    break
                except Exception:
                    pass

    def get_emoji(self, word: str, is_figurative: bool = False, pos: str = None) -> str:
        word_lower = word.lower().strip()
        
        # Helper check
        def lookup(w):
            if w in self.figurative_mappings and is_figurative:
                return self.figurative_mappings[w]
            if w in self.literal_mappings and not is_figurative:
                return self.literal_mappings[w]
            if w in self.emoji_dict:
                return self.emoji_dict[w]
            if w in self.fallback_dict:
                return self.fallback_dict[w]
            w_under = w.replace(" ", "_")
            if w_under in self.fallback_dict:
                return self.fallback_dict[w_under]
            return None

        # 1. Direct match
        res = lookup(word_lower)
        if res:
            return res

        # 2. Suffix stripping
        for suffix in ['ed', 'ing', 's', 'es', 'er', 'or', 'ly']:
            if word_lower.endswith(suffix):
                stem = word_lower[:-len(suffix)]
                if len(stem) >= 3:
                    res = lookup(stem)
                    if res:
                        return res

        # 3. WordNet Semantic Matching
        try:
            from nltk.corpus import wordnet as wn
            synsets = wn.synsets(word_lower)
            # Try synonym lemmas first
            for syn in synsets:
                for lemma in syn.lemmas():
                    name = lemma.name().lower().replace('_', ' ')
                    res = lookup(name)
                    if res:
                        return res
            # Try hypernym lemmas
            for syn in synsets:
                for hyper in syn.hypernyms():
                    for lemma in hyper.lemmas():
                        name = lemma.name().lower().replace('_', ' ')
                        res = lookup(name)
                        if res:
                            return res
        except Exception:
            pass

        # 4. POS-based select fallback
        if pos:
            pos_upper = pos.upper()
            if pos_upper == "VERB":
                return "🏃"
            elif pos_upper == "NOUN":
                return "📦"
            elif pos_upper == "ADJ":
                return "✨"
            elif pos_upper == "ADV":
                return "⚡"
                
        return "❓"

# ---------------------------------------------------------------------------
# ImageLinkGenerator Class
# ---------------------------------------------------------------------------
class ImageLinkGenerator:
    def __init__(self):
        pass

    def validate_url(self, url: str) -> bool:
        """Validate URL is reachable using a fast head request."""
        try:
            headers = {"User-Agent": "SignDecoderVisualLinker/1.0 (contact: support@signdecoder.org)"}
            response = requests.head(url, headers=headers, timeout=0.3, allow_redirects=True)
            return response.status_code == 200
        except Exception:
            return False

    def get_image_url(self, word: str, is_figurative: bool = False) -> str:
        """Find the best image url according to priority order."""
        word_clean = word.lower().strip()
        word_query = urllib.parse.quote_plus(word_clean)
        # Curated high-quality Unsplash image URLs for common simplified words
        common_images = {
            "doctor": "https://images.unsplash.com/photo-1622253692010-333f2da6031d?auto=format&fit=crop&w=400&q=80",
            "sick": "https://images.unsplash.com/photo-1584824486509-112e4181ff6b?auto=format&fit=crop&w=400&q=80",
            "ill": "https://images.unsplash.com/photo-1584824486509-112e4181ff6b?auto=format&fit=crop&w=400&q=80",
            "tired": "https://images.unsplash.com/photo-1515003197210-e0cd71810b5f?auto=format&fit=crop&w=400&q=80",
            "happy": "https://images.unsplash.com/photo-1507679799987-c73779587ccf?auto=format&fit=crop&w=400&q=80",
            "sad": "https://images.unsplash.com/photo-1516585424559-8bab1162cc7e?auto=format&fit=crop&w=400&q=80",
            "bought": "https://images.unsplash.com/photo-1542838132-92c53300491e?auto=format&fit=crop&w=400&q=80",
            "school": "https://images.unsplash.com/photo-1580582932707-520aed937b7b?auto=format&fit=crop&w=400&q=80",
            "home": "https://images.unsplash.com/photo-1513694203232-719a280e022f?auto=format&fit=crop&w=400&q=80",
            "house": "https://images.unsplash.com/photo-1513694203232-719a280e022f?auto=format&fit=crop&w=400&q=80",
            "medicine": "https://images.unsplash.com/photo-1471864190281-a93a3070b6de?auto=format&fit=crop&w=400&q=80",
            "hospital": "https://images.unsplash.com/photo-1586773860418-d37222d8fce2?auto=format&fit=crop&w=400&q=80",
            "died": "https://images.unsplash.com/photo-1534438327276-14e5300c3a48?auto=format&fit=crop&w=400&q=80",
            "death": "https://images.unsplash.com/photo-1534438327276-14e5300c3a48?auto=format&fit=crop&w=400&q=80",
            "courage": "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?auto=format&fit=crop&w=400&q=80",
            "patience": "https://images.unsplash.com/photo-1518241353330-0f7941c2d9b5?auto=format&fit=crop&w=400&q=80",
            "government": "https://images.unsplash.com/photo-1541872703-74c5e44368f9?auto=format&fit=crop&w=400&q=80",
            "law": "https://images.unsplash.com/photo-1589829545856-d10d557cf95f?auto=format&fit=crop&w=400&q=80",
            "patient": "https://images.unsplash.com/photo-1516549655169-df83a0774514?auto=format&fit=crop&w=400&q=80",
            "treat": "https://images.unsplash.com/photo-1576091160550-2173dba999ef?auto=format&fit=crop&w=400&q=80",
            "treated": "https://images.unsplash.com/photo-1576091160550-2173dba999ef?auto=format&fit=crop&w=400&q=80",
            "treatment": "https://images.unsplash.com/photo-1576091160550-2173dba999ef?auto=format&fit=crop&w=400&q=80",
            "help": "https://images.unsplash.com/photo-1469571486117-4ac50143d2d3?auto=format&fit=crop&w=400&q=80",
            "helper": "https://images.unsplash.com/photo-1469571486117-4ac50143d2d3?auto=format&fit=crop&w=400&q=80",
            "work": "https://images.unsplash.com/photo-1497366216548-37526070297c?auto=format&fit=crop&w=400&q=80",
            "job": "https://images.unsplash.com/photo-1497366216548-37526070297c?auto=format&fit=crop&w=400&q=80",
            "worker": "https://images.unsplash.com/photo-1521791136368-1a4ec8cc8f2c?auto=format&fit=crop&w=400&q=80",
            "family": "https://images.unsplash.com/photo-1542038784456-1ea8e935640e?auto=format&fit=crop&w=400&q=80",
            "friend": "https://images.unsplash.com/photo-1511632765486-a01980e01a18?auto=format&fit=crop&w=400&q=80",
            "happy": "https://images.unsplash.com/photo-1507679799987-c73779587ccf?auto=format&fit=crop&w=400&q=80",
            "car": "https://images.unsplash.com/photo-1494976388531-d1058094e2fd?auto=format&fit=crop&w=400&q=80",
            "vehicle": "https://images.unsplash.com/photo-1494976388531-d1058094e2fd?auto=format&fit=crop&w=400&q=80",
            "play": "https://images.unsplash.com/photo-1531415074968-036ba1b575da?auto=format&fit=crop&w=400&q=80",
            "playing": "https://images.unsplash.com/photo-1531415074968-036ba1b575da?auto=format&fit=crop&w=400&q=80",
        }

        if word_clean in common_images:
            return common_images[word_clean]

        if is_figurative:
            return f"https://images.unsplash.com/featured/400x300/?concept,{word_query}"
        else:
            return f"https://images.unsplash.com/featured/400x300/?{word_query}"

    def get_all_links(self, word: str, is_figurative: bool = False) -> dict:
        word_clean = word.lower().strip()
        word_query = urllib.parse.quote_plus(word_clean)
        
        wikimedia = f"https://commons.wikimedia.org/wiki/File:{word_clean.capitalize()}.jpg"
        unsplash = self.get_image_url(word, is_figurative)
        google_images = f"https://www.google.com/search?q={word_query}+meaning&tbm=isch"
        merriam_webster = f"https://www.merriam-webster.com/dictionary/{word_clean}"
        oxford = f"https://www.oxfordlearnersdictionaries.com/definition/english/{word_clean}"
        wikipedia = f"https://en.wikipedia.org/wiki/{word_clean.capitalize()}"
        
        primary = unsplash
        
        return {
            "wikimedia": wikimedia,
            "unsplash": unsplash,
            "google_images": google_images,
            "merriam_webster": merriam_webster,
            "oxford": oxford,
            "wikipedia": wikipedia,
            "primary": primary
        }

# ---------------------------------------------------------------------------
# OutputFormatter Class
# ---------------------------------------------------------------------------
class OutputFormatter:
    @staticmethod
    def to_terminal(original: str, simplified: str, changes: List[Dict[str, Any]]) -> str:
        lines = []
        lines.append("════════════════════════════════════════")
        lines.append("LEXICAL SIMPLIFICATION RESULT")
        lines.append("════════════════════════════════════════")
        lines.append("")
        lines.append("ORIGINAL:")
        lines.append(f'"{original}"')
        lines.append("")
        lines.append("SIMPLIFIED:")
        lines.append(f'"{simplified}"')
        lines.append("")
        lines.append("════════════════════════════════════════")
        lines.append("WORD EXPLANATIONS")
        lines.append("════════════════════════════════════════")
        lines.append("")
        
        for idx, change in enumerate(changes, 1):
            orig_word = change["original_word"]
            simp_word = change["simplified_word"]
            emoji = change["emoji"]
            defn = change["definition"]
            ex = change["example"]
            unsplash_img = change["images"]["unsplash"]
            mw_dict = change["links"]["merriam_webster"]
            
            lines.append(f"{orig_word} → {simp_word}")
            lines.append("─" * len(f"{orig_word} → {simp_word}"))
            lines.append(f"Emoji:       {emoji}")
            lines.append(f"Meaning:     {defn}")
            lines.append(f"Example:     {ex}")
            lines.append(f"Image Link:  [Click Here - {unsplash_img}]")
            lines.append(f"Dictionary:  [Click Here - {mw_dict}]")
            lines.append("")
            
        lines.append("════════════════════════════════════════")
        return "\n".join(lines)

    @staticmethod
    def to_html(original: str, simplified: str, changes: List[Dict[str, Any]]) -> str:
        html = []
        html.append('<div class="simplification-result">')
        
        html.append('  <div class="original">')
        html.append('    <h3>Original Sentence</h3>')
        html.append(f'    <p>"{original}"</p>')
        html.append('  </div>')

        html.append('  <div class="simplified">')
        html.append('    <h3>Simplified Sentence</h3>')
        html.append(f'    <p>"{simplified}"</p>')
        html.append('  </div>')

        html.append('  <div class="word-changes">')
        html.append('    <h3>Word Changes</h3>')
        
        for change in changes:
            orig = change["original_word"]
            simp = change["simplified_word"]
            emoji = change["emoji"]
            defn = change["definition"]
            ex = change["example"]
            primary_img = change["images"]["primary"]
            mw_dict = change["links"]["merriam_webster"]
            pron = change.get("pronunciation", "")
            
            html.append('    <div class="word-card">')
            html.append(f'      <span class="original-word">{orig}</span>')
            html.append('      <span class="arrow">→</span>')
            html.append(f'      <span class="simple-word">{simp}</span>')
            html.append(f'      <span class="emoji">{emoji}</span>')
            html.append('      <div class="links">')
            html.append(f'        <a href="{primary_img}" target="_blank" class="image-link">📸 See Image</a>')
            html.append(f'        <a href="{mw_dict}" target="_blank" class="dict-link">📖 Definition</a>')
            html.append('      </div>')
            html.append('      <div class="tooltip">')
            html.append(f'        <img src="{primary_img}" alt="{simp}" width="200" />')
            html.append(f'        <h4 style="margin: 4px 0;">{simp} {emoji}</h4>')
            if pron:
                html.append(f'        <p class="pron"><i>{pron}</i></p>')
            html.append(f'        <p class="defn"><b>Meaning:</b> {defn}</p>')
            html.append(f'        <p class="ex"><b>Example:</b> {ex}</p>')
            html.append('      </div>')
            html.append('    </div>')
            
        html.append('  </div>')
        html.append('</div>')
        return "\n".join(html)

    @staticmethod
    def to_json(original: str, simplified: str, changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "original": original,
            "simplified": simplified,
            "changes": changes
        }

# ---------------------------------------------------------------------------
# VisualWordLinker Class
# ---------------------------------------------------------------------------
class VisualWordLinker:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or VISUAL_CONFIG
        self.emoji_mapper = EmojiMapper()
        self.image_generator = ImageLinkGenerator()
        
        # Load local dictionary cache to support fast / offline responses
        self.cache_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dictionary_cache.json")
        self.dict_cache = {}
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.dict_cache = json.load(f)
            except Exception:
                pass

    def _save_cache(self):
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.dict_cache, f, indent=2)
        except Exception:
            pass

    def get_emoji(self, word: str, sense: str = None, pos: str = None, is_figurative: bool = False) -> str:
        return self.emoji_mapper.get_emoji(word, is_figurative, pos)

    def get_image_url(self, word: str, sense: str = None, is_figurative: bool = False) -> str:
        return self.image_generator.get_image_url(word, is_figurative)

    def get_definition(self, word: str, pos: str = None) -> str:
        info = self.get_word_details(word, pos)
        return info.get("definition", "Meaning not found.")

    def get_pronunciation(self, word: str) -> str:
        info = self.get_word_details(word)
        return info.get("pronunciation", "")

    def get_audio_link(self, word: str) -> str:
        return f"https://www.merriam-webster.com/dictionary/{word}#audio"

    def get_word_details(self, word: str, pos: str = None) -> Dict[str, Any]:
        word_clean = word.lower().strip()
        
        # 1. Check local cache
        if word_clean in self.dict_cache:
            return self.dict_cache[word_clean]
            
        # 2. Try online Free Dictionary API
        api_url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word_clean}"
        try:
            response = requests.get(api_url, timeout=0.8) # tight timeout
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    entry = data[0]
                    phonetic = entry.get("phonetic", "")
                    if not phonetic and entry.get("phonetics"):
                        phonetic = entry["phonetics"][0].get("text", "")
                        
                    definition = ""
                    example = ""
                    related = []
                    
                    for meaning in entry.get("meanings", []):
                        for definition_obj in meaning.get("definitions", []):
                            if not definition:
                                definition = definition_obj.get("definition", "")
                            if not example:
                                example = definition_obj.get("example", "")
                        for synonym in meaning.get("synonyms", []):
                            related.append(synonym)
                            
                    if not example:
                        example = f"The {word_clean} was observed."
                        
                    result = {
                        "definition": definition,
                        "pronunciation": phonetic,
                        "example": example,
                        "related": related[:3]
                    }
                    self.dict_cache[word_clean] = result
                    self._save_cache()
                    return result
        except Exception:
            pass
            
        # 3. Fallback to local NLTK WordNet (Works offline!)
        try:
            pos_map = {
                'NOUN': wn.NOUN,
                'VERB': wn.VERB,
                'ADJ': wn.ADJ,
                'ADV': wn.ADV,
                'PROPN': wn.NOUN
            }
            wn_pos = pos_map.get(pos.upper()) if pos else None
            synsets = wn.synsets(word_clean, pos=wn_pos) if wn_pos else wn.synsets(word_clean)
            if not synsets:
                synsets = wn.synsets(word_clean)
                
            if synsets:
                syn = synsets[0]
                definition = syn.definition()
                examples = syn.examples()
                example = examples[0] if examples else f"The {word_clean} was noticed."
                related = [l.name().replace('_', ' ') for l in syn.lemmas() if l.name().lower() != word_clean]
                
                # Approximate pronunciation breakdown (syllables)
                pronunciation = f"/{word_clean}/"
                
                result = {
                    "definition": definition.capitalize(),
                    "pronunciation": pronunciation,
                    "example": example.capitalize(),
                    "related": related[:3]
                }
                self.dict_cache[word_clean] = result
                self._save_cache()
                return result
        except Exception:
            pass
            
        # Hardcoded fallback
        return {
            "definition": "Definition unavailable.",
            "pronunciation": f"/{word_clean}/",
            "example": f"Please refer to the dictionary for {word_clean}.",
            "related": []
        }

    def process_substitutions(self, original_sentence: str, simplified_sentence: str, replacements: Dict[str, str], word_info_list: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Processes a list of word changes/substitutions, fetches images, definitions, 
        and maps emojis, returning standard structured details.
        """
        changes = []
        for index, (orig, simp) in enumerate(replacements.items()):
            # Detect category or pos from word_info_list if provided
            pos = None
            is_figurative = False
            
            # Simple heuristic detection if info list not given
            if word_info_list:
                for info in word_info_list:
                    if info.get("word") == orig:
                        pos = info.get("pos")
                        is_figurative = info.get("category", "") in ("Idiom", "Metaphor")
                        break
            
            # Let's perform context detection based on word lists if not found
            if not is_figurative:
                # E.g. check if part of standard idiom replacements
                is_figurative = any(idiom in original_sentence.lower() for idiom in ["kick the bucket", "under the weather", "spill the beans", "hit the nail", "bite the bullet", "let the cat", "burn the midnight", "beat around the bush", "break the ice", "arm and a leg"])

            # 1. Get Emoji (context aware)
            emoji = self.get_emoji(simp, is_figurative, pos)
            
            # 2. Get Details (cached dictionary)
            details = self.get_word_details(simp, pos)
            
            # 3. Get Images & Links
            images = self.image_generator.get_all_links(simp, is_figurative)
            
            # Complexity score reduction estimation
            # Zipf complexity (approximate)
            import wordfreq
            orig_zipf = wordfreq.zipf_frequency(orig.lower(), 'en')
            simp_zipf = wordfreq.zipf_frequency(simp.lower(), 'en')
            
            # normalize to [0, 1] range representing difficulty (lower zipf = higher complexity)
            orig_score = max(0.0, min(1.0, (8.0 - orig_zipf) / 8.0)) if orig_zipf > 0 else 0.8
            simp_score = max(0.0, min(1.0, (8.0 - simp_zipf) / 8.0)) if simp_zipf > 0 else 0.4
            reduction = orig_score - simp_score
            
            changes.append({
                "original_word": orig,
                "simplified_word": simp,
                "position": index + 1,
                "emoji": emoji,
                "definition": details.get("definition", "Definition unavailable."),
                "pronunciation": details.get("pronunciation", f"/{simp}/"),
                "example": details.get("example", f"The {simp} was observed."),
                "images": {
                    "wikimedia": images["wikimedia"],
                    "unsplash": images["unsplash"],
                    "primary": images["primary"]
                },
                "links": {
                    "merriam_webster": images["merriam_webster"],
                    "oxford": images["oxford"],
                    "google_images": images["google_images"],
                    "wikipedia": images["wikipedia"]
                },
                "complexity_score": {
                    "original": round(orig_score, 2),
                    "simplified": round(simp_score, 2),
                    "reduction": round(reduction, 2)
                }
            })
            
        # Calculate sentence-level complexity
        import wordfreq
        def calc_sentence_difficulty(sent: str) -> float:
            words = [w.strip(".,!?;:\'\"").lower() for w in sent.split() if w.strip(".,!?;:\'\"").isalpha()]
            if not words:
                return 0.5
            total_diff = 0.0
            for w in words:
                zipf = wordfreq.zipf_frequency(w, 'en')
                diff = max(0.0, min(1.0, (8.0 - zipf) / 8.0)) if zipf > 0 else 0.5
                total_diff += diff
            return total_diff / len(words)

        orig_sentence_diff = calc_sentence_difficulty(original_sentence)
        simp_sentence_diff = calc_sentence_difficulty(simplified_sentence)
        sentence_reduction = orig_sentence_diff - simp_sentence_diff

        sentence_complexity = {
            "original": round(orig_sentence_diff, 2),
            "simplified": round(simp_sentence_diff, 2),
            "reduction": round(sentence_reduction, 2)
        }

        # Calculate sentence theme query and featured image link
        cleaned_sent = simplified_sentence.lower().strip()
        stop_words = {"the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "to", "of", "in", "on", "at", "by", "for", "with", "about", "as", "this", "that", "it", "they", "he", "she", "you", "we", "i", "my", "your", "his", "her", "their", "our", "been", "have", "has", "had", "do", "does", "did", "will", "would", "shall", "should", "can", "could", "may", "might", "must"}
        words_sent = [w.strip(".,!?;:\'\"") for w in cleaned_sent.split() if w.strip(".,!?;:\'\"").isalpha()]
        keywords_sent = [w for w in words_sent if w not in stop_words]
        sentence_theme = " ".join(keywords_sent[:3]) if keywords_sent else "concept"
        sentence_image = self.image_generator.get_image_url(sentence_theme)
        
        sentence_visuals = {
            "theme": sentence_theme,
            "image": sentence_image,
            "google_images": f"https://www.google.com/search?q={urllib.parse.quote_plus(sentence_theme)}+meaning&tbm=isch",
            "wikipedia": f"https://en.wikipedia.org/wiki/{urllib.parse.quote_plus(sentence_theme)}"
        }

        return {
            "original": original_sentence,
            "simplified": simplified_sentence,
            "changes": changes,
            "sentence_complexity": sentence_complexity,
            "sentence_visuals": sentence_visuals
        }

    def format_terminal_output(self, result: Dict[str, Any]) -> str:
        return OutputFormatter.to_terminal(result["original"], result["simplified"], result["changes"])

    def format_html_output(self, result: Dict[str, Any]) -> str:
        return OutputFormatter.to_html(result["original"], result["simplified"], result["changes"])

    def format_json_output(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return OutputFormatter.to_json(result["original"], result["simplified"], result["changes"])

"""
category_matcher_engine.py
──────────────────────────
Pure-Python matching engine extracted from CategoryMatcher.
Now wired to Firebase Firestore for persistent cloud storage.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import re
import warnings
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

# ── Streamlit and Firebase Imports ───────────────────────────────────────────
import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore

# ── scikit-learn (optional — falls back to manual TF-IDF) ────────────────────
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn(
        "scikit-learn not found. Install with: pip install scikit-learn\n"
        "Category matcher will fall back to the built-in similarity engine.",
        RuntimeWarning, stacklevel=2,
    )


class CategoryMatcherEngine:
    """
    Self-contained category matching engine.
    Instantiate once; call build_tfidf_index() with your categories list,
    then call get_category_with_fallback() for every product.
    """

    # ── Paths for persisted learning data ────────────────────────────────────
    _BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
    _SK_MODEL_PATH    = os.path.join(_BASE_DIR, "sk_model.pkl")

    def __init__(self):
        # Initialize Firebase first
        self.db_client = self._init_firebase()
        
        # Load directly from the cloud instead of a local JSON file
        self.learning_db: dict[str, str] = self.load_learning_db()
        self._last_categories_list: list[str] = []

        # TF-IDF / sklearn state
        self._tfidf_built       = False
        self._domain_indexes    = {}
        self._domain_index_cats: list[str] = []

        self._sk_global_vec  = None
        self._sk_global_mat  = None
        self._sk_domain_vecs: dict = {}
        self._sk_built       = False
        self._sk_cats: list[str] = []

        self._sk_clf         = None
        self._sk_clf_vec     = None
        self._sk_le          = None
        self._sk_clf_trained = False

        self._load_sklearn_model()

    def _init_firebase(self):
        """Initialize Firebase securely using Streamlit Secrets. Prevents double-init errors."""
        if not firebase_admin._apps:
            firebase_secrets = dict(st.secrets["firebase"])
            cred = credentials.Certificate(firebase_secrets)
            firebase_admin.initialize_app(cred)
        return firestore.client()

    _SEED_CORRECTIONS = {
        # Ladies nightwear → Women's Fashion Sleep & Lounge (not Books, Baby, Auto, Home)
        'beautiful nightwear new design for ladies':
            "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Sets",
        'ladies nightwear top and short':
            "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Sets",
        'ladies nightwear top and trouser':
            "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Sets",
        'nightwear new design for ladies':
            "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Sets",
        'beautiful nightwear for ladies':
            "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Sets",
        'sexy nightwear for ladies':
            "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Sets",
        'lovely nightwear for ladies':
            "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Sets",
        'classy nightwear for ladies':
            "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Sets",
        'elegant nightwear for ladies':
            "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Sets",
        'ladies sexy nightwear':
            "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Sets",
        'ladies classy nightwear':
            "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Sets",
        'ladies lovely nightwear':
            "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Sets",
        'ladies elegant nightwear':
            "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Sets",
        'sleepwear for ladies top and short':
            "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Sets",
        'nightwear for women different colours':
            "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Sets",
        'ladies beautiful and lovely nightwear':
            "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Sets",
        'ladies new design nightwear':
            "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Sets",
        'ladies nightwear new design':
            "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Sets",
        'new design sexy nightwear for ladies':
            "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Sets",
        # Ladies singlet / tops → Women's Tops (not Baby, Industrial, Home)
        '3pcs ladies quality stretchable tops':
            "Fashion / Women's Fashion / Clothing / Tops & Tees",
        '3pcs ladies quality stretchable camisole':
            "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Lingerie / Camisoles & Tanks",
        'ladies singlet top multicolor 3pcs':
            "Fashion / Women's Fashion / Clothing / Tops & Tees",
        'ladies singlet top colur differs':
            "Fashion / Women's Fashion / Clothing / Tops & Tees",
        # Ladies pants / cotton pant → Women's Lingerie/Panties (not Men's, Industrial)
        'sexy ladies cotton pant 3pcs':
            "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Lingerie / Panties",
        'sexy ladies cotton pant 6pcs':
            "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Lingerie / Panties",
        'ladies sexy tight 3pcs':
            "Fashion / Women's Fashion / Clothing / Leggings",
        '3pic beautiful ladies sexy tight':
            "Fashion / Women's Fashion / Clothing / Leggings",
        # Bumper tight → Leggings (not Automobile)
        'bumper tight for ladies different 3pcs':
            "Fashion / Women's Fashion / Clothing / Leggings",
        'bumper tight for ladies different colours 3pcs':
            "Fashion / Women's Fashion / Clothing / Leggings",
        'bumper tight for ladies black 2pcs':
            "Fashion / Women's Fashion / Clothing / Leggings",
        # Ladies casual top → Women's Clothing (not Men's)
        'ladies casual top and pants set':
            "Fashion / Women's Fashion / Clothing / Pants",
        # Ladies gown → Women's Dresses (not Home, Baby, Industrial)
        'ladies sexy gown wine':
            "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Nightgowns & Sleepshirts",
        'ladies beautiful characters gowns':
            "Fashion / Women's Fashion / Clothing / Dresses",
        'sexy nightgown for ladies':
            "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Nightgowns & Sleepshirts",
        # Ladies underwear/boyshorts → Women's Panties (not Industrial, Men's)
        '6pcs women strechy boyshorts':
            "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Lingerie / Panties",
        '6 in 1 set of sexy ladies pants':
            "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Lingerie / Panties",
        '6pcs mixture of body colours ladies tight underwears':
            "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Lingerie / Panties",
        'ladies underwear woman sexy lace transparent low-rise':
            "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Lingerie / Panties",
        # Cargo jean for ladies → Women's Jeans (not Men's)
        'boyfriend cargo jean for ladies':
            "Fashion / Women's Fashion / Clothing / Jeans",
        # G-string → Women's Panties (not Fashion High Heels)
        'sixzy classic ladies g string 3pic':
            "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Lingerie / Panties",
        # Ladies wear short → Women's Sleep (not Automobile Engine Block)
        'ladies wear top and short':
            "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Sets",
        'ladies top and short nightwear':
            "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Sets",
        'ladies sexy top and short':
            "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Sets",
        'ladies night wear top and short':
            "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Sets",
        'sexy night wear for ladies top and short':
            "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Sets",
        'ladies wear top and trouser':
            "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Sets",
        # High quality ladies pant/crop top → Women's Clothing (not Industrial Drill Bits)
        'high quality ladies long length baggy cut pant':
            "Fashion / Women's Fashion / Clothing / Pants",
        # Night lights confusion
        'sexy night wear for ladies':
            "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Sets",
        'ladies night wear':
            "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Sets",
        'ladies beautiful night gown':
            "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Nightgowns & Sleepshirts",
        'ladies night gown':
            "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Nightgowns & Sleepshirts",
        'ladies night wear gown':
            "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Nightgowns & Sleepshirts",
        # G-string lace → Women's Panties (not Home Office Lace/Trim)
        'g-string ladies underwear pants cotton lace':
            "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Lingerie / Panties",
        # 3-in-1 ladies (not Baby Car Seats)
        'beautiful nightwear for ladies 3 in 1':
            "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Sets",
        '3 in 1 nightwear for ladies':
            "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Sets",
        '3 in 1 ladies beautiful bratop':
            "Fashion / Women's Fashion / Underwear & Sleepwear / Bra",
        '3 in 1 classy nightwear for ladies':
            "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Sets",
        '2 in 1 nightwear for ladies':
            "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Sets",
        '2 in 1 beautiful nightwear for ladies':
            "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Sets",
    }

    _DOMAIN_KEYWORDS = {
        'Phones & Tablets': [
            'mobile phone','smartphone','android phone','iphone','samsung galaxy',
            'tecno phone','infinix phone','infinix hot','infinix note','itel phone',
            'redmi note','xiaomi poco','oppo reno','vivo phone','realme phone',
            'nokia phone','tablet pc','ipad pro','sim card tray','power bank charger',
            'powerbank wireless','screen protector glass','tempered glass screen',
            'phone case cover','type c cable','lightning cable','selfie stick phone',
            'phone holder car','pop socket grip','otg adapter','phone stand desk',
            'phone charger fast','wireless earbuds phone','samsung a','samsung m',
            'galaxy s','galaxy a','iphone pro','iphone plus','infinix','tecno camon',
        ],
        'Computing': [
            'laptop computer','notebook computer','chromebook','desktop computer',
            'all in one pc','computer monitor','pc keyboard','computer mouse',
            'wireless mouse','usb hub','hdmi cable','vga cable','external hard drive',
            'ssd drive','flash drive','pendrive','usb drive','memory card','sd card',
            'wifi router','internet router','network switch','printer ink',
            'laptop bag','laptop sleeve','laptop backpack','cooling pad',
            'ups battery backup','computer speaker','graphics card','processor chip',
            'desktop pc','computer desk setup',
        ],
        'Electronics': [
            'smart tv','led tv','oled tv','4k tv','flat screen tv','plasma tv',
            'soundbar','sound bar','home theater system','bluetooth speaker',
            'portable speaker','wireless headphones','noise cancelling headphones',
            'security camera','cctv camera','ip camera','action camera',
            'digital camera','dslr camera','mirrorless camera','drone camera',
            'streaming device','android tv box','dvd player','blu ray player',
            'walkie talkie','gps tracker','solar panel inverter',
            'smart bulb','smart plug','smart home device','television',
        ],
        'Fashion': [
            # Women / Ladies — explicit signals (high-priority, long phrases first)
            'ladies nightwear','ladies sleepwear','ladies night wear','ladies nightgown',
            'nightwear for ladies','sleepwear for ladies','nightwear for women',
            'ladies top and short','ladies top and trouser','ladies top and pant',
            'ladies casual top','ladies sexy top','ladies singlet top','ladies wear',
            'ladies panties','ladies underwear','ladies bra','ladies lingerie',
            'ladies tight','ladies camisole','ladies pant','ladies gown','ladies dress',
            'ladies skirt','ladies jean','ladies legging','ladies blouse','ladies shirt',
            'ladies shorts','ladies kaftan','ladies abaya','ladies kimono',
            'for ladies','for women','women nightwear','women sleepwear','women gown',
            'women dress','women top','women shirt','women blouse','women pant',
            'women jean','women legging','women bra','women underwear','women panties',
            'women lingerie','women camisole','women kaftan','sexy ladies','sexy women',
            'female pant','female tight','female camisole','female panties',
            'beautiful ladies','classy ladies','lovely ladies','elegant ladies',
            'bumper tight','biker short','boyshorts','g-string ladies','gstring ladies',
            'ladies condom','ladies nika','ladies menstrual','ladies cotton pant',
            'ladies sexy pant','ladies quality','ladies beautiful','ladies lovely',
            'ladies classy','ladies elegant','ladies unique','ladies new design',
            'ladies stretchable','ladies push up','ladies breathable',
            # Men
            'men trouser','men jeans','men shorts','men jacket','men blazer',
            'men suit','men hoodie','men boxers','men sneakers','men loafers',
            'polo shirt','boxer shorts',
            # General fashion
            'agbada','senator suit','kaftan','ankara fabric','lace fabric',
            'aso oke','gele','traditional wear','african wear','native wear',
            'swimwear','bikini','high heels','flat shoes','ankle boots',
            'ladies handbag','women purse','clutch bag','fashion wallet',
            'ladies belt','fashion cap','ladies sunglasses','necklace jewelry',
            'ladies bracelet','women earrings','ring jewelry','hair wig',
            'hair extension','hair weave','nightwear','pyjamas','sleepwear',
            'shapewear','waist trainer','body shaper','school uniform',
            'children clothing','kids wear',
        ],
        'Health & Beauty': [
            'face serum','face moisturizer','face cream','body lotion','sunscreen spf',
            'face wash','toner skincare','foundation makeup','bb cream','concealer',
            'lipstick','lip gloss','eyeshadow palette','mascara','eyeliner',
            'blush makeup','makeup brush set','nail polish','nail gel kit',
            'perfume women','cologne men','deodorant roll on','body spray',
            'hair shampoo','hair conditioner','hair growth oil','beard kit',
            'hair dryer','flat iron hair','hair straightener','curling iron',
            'hair clipper','electric trimmer','electric shaver','waxing kit',
            'vitamin supplement','whey protein','blood pressure monitor',
            'glucose meter','glucometer','pulse oximeter','digital thermometer',
            'first aid supplies','sanitary pad','menstrual pad','tampon',
            'feminine wash','condom','massage gun','slimming supplement',
            'collagen supplement','human hair wig','hair extension wig',
        ],
        'Home & Office': [
            'sofa set','couch living room','bed frame','mattress bed','pillow sleep',
            'bedsheet set','duvet cover','wardrobe closet','dressing table',
            'bookshelf wood','dining table set','coffee table','tv stand','office desk',
            'office chair','window curtain','venetian blind','carpet rug',
            'wall art decor','wall mirror','picture frame','wall clock',
            'scent diffuser','desk lamp','floor lamp','ceiling fan','standing fan',
            'table fan','air conditioner split','refrigerator fridge',
            'washing machine laundry','microwave oven','electric blender',
            'toaster oven','electric kettle','cooking pot set','frying pan',
            'cutlery set','dinner plates','storage box organizer','laundry basket',
            'clothes iron steam','vacuum cleaner','cleaning mop',
            'office stationery','office pen set','paper shredder','laminator machine',
            'whiteboard office','standing desk adjustable','sewing machine',
        ],
        'Automobile': [
            'car wax polish','car wash kit','car shampoo','car seat cover',
            'car floor mat','car air freshener','car phone mount','dash cam',
            'car stereo radio','car alarm system','car tyre','car tire',
            'wheel rim','brake pad','brake disc','engine oil','motor oil',
            'gear oil transmission','car battery','spark plug','car headlight',
            'tail light led','parking sensor reverse','car jack','tire inflator',
            'jump starter','car cover protection','windshield wiper',
            'motorcycle helmet','bike chain oil',
        ],
        'Baby Products': [
            'baby formula milk','infant formula','baby food puree','baby cereal',
            'diapers baby','baby nappy','baby wipes','baby lotion','baby oil',
            'baby powder','baby shampoo','baby bath wash','baby bottle feeding',
            'pacifier dummy','baby monitor','baby carrier wrap','baby stroller',
            'pram pushchair','baby cot','baby crib','high chair feeding',
            'baby walker ring','baby bouncer','baby swing','teether toy',
            'baby rattle','play mat baby','baby blanket swaddle','baby romper',
        ],
        'Sporting Goods': [
            'football soccer','basketball hoop','tennis racket','badminton racket',
            'volleyball net','cricket bat','golf club iron','treadmill running',
            'exercise bike stationary','rowing machine','dumbbell set',
            'barbell weight','kettlebell','resistance band set','pull up bar',
            'yoga mat thick','jump rope skipping','boxing gloves','punching bag',
            'mma gear','bicycle mountain','road bike','cycling helmet',
            'swim goggles','fishing rod','fishing reel','camping tent',
            'sleeping bag','hiking boots','trekking pole','football boots',
            'gym bag sports','fitness tracker band','sport water bottle',
            'running shoes','sport shoes',
        ],
        'Toys & Games': [
            'lego building','toy blocks','action figure toy','barbie doll',
            'remote control car toy','rc remote control','rc toy car','rc toy helicopter',
            'toy truck kids','board game family','chess set board','jigsaw puzzle',
            'card game','teddy bear plush','stuffed animal toy','water gun toy',
            'nerf blaster','kids scooter','children bicycle','trampoline kids',
            'swing set park','playdoh clay','slime toy kit','bubble machine',
            'yo yo toy','toy kitchen set','pretend play kids',
        ],
        'Gaming': [
            'playstation 5','ps5 console','ps4 console','xbox series','nintendo switch',
            'gaming console','gaming controller','wireless gamepad',
            'gaming headset','gaming chair','gaming keyboard rgb',
            'gaming mouse','gaming monitor','game cartridge','pc gaming setup',
            'gaming laptop','game capture card','retro game console',
        ],
        'Pet Supplies': [
            'dog food dry','cat food wet','fish food flakes','bird seed food',
            'pet food kibble','kibble dog','dog kibble','adult dog food',
            'puppy food','kitten food','dog treat','cat treat','dog collar leash',
            'pet carrier bag','bird cage','fish tank aquarium',
            'pet bed cushion','dog toy ball','cat toy feather','pet shampoo',
            'flea tick treatment','cat litter box','litter sand',
            'pet water bowl','dog grooming kit','premium pet food',
        ],
        'Garden & Outdoors': [
            'garden hose pipe','watering can','garden sprinkler','garden spade',
            'pruning shears','lawn mower','grass cutter','hedge trimmer',
            'plant pot flower','planter box','garden soil','fertilizer bag',
            'pesticide spray','outdoor furniture set','garden chair patio',
            'bbq grill charcoal','barbecue grill','fire pit outdoor',
            'camping stove','solar garden light','bird feeder outdoor',
        ],
        'Grocery': [
            'scouring pad','cleaning sponge','dishwashing liquid','trash bag',
            'paper towel','toilet paper roll','aluminum foil','food storage bag',
            'zip lock bag','disposable plate','disposable cup','plastic wrap',
            'drain cleaner','wood polish','fabric cleaner','hookah shisha',
            'cigar accessories','tobacco pipe','cigarette holder',
        ],
        'Books, Movies and Music': [
            'fiction novel book','textbook academic','dictionary language',
            'self help book','biography book','cookbook recipe',
            'dvd movie film','blu ray disc','music album cd','vinyl record',
            'kindle ebook',
        ],
        'Musical Instruments': [
            'acoustic guitar','electric guitar','bass guitar','guitar strings',
            'piano keyboard instrument','digital piano','drum set kit',
            'violin instrument','trumpet brass','saxophone wind','flute instrument',
            'harmonica mouth','microphone vocal','audio mixer dj',
            'audio interface recording','guitar amplifier','dj turntable',
            'instrument tuner','guitar capo',
        ],
        'Industrial & Scientific': [
            'digital multimeter','soldering iron station','voltage tester pen',
            'clamp meter current','power drill cordless','cordless drill',
            'angle grinder disk','jigsaw machine','circular saw','impact drill',
            'hammer drill','screwdriver set tool','wrench spanner set','pliers tool set',
            'safety helmet hard hat','safety vest reflective','welding machine',
            'welding rod electrode','air compressor tank','pressure washer pump',
            'measuring tape roll','spirit level','vernier caliper',
            'fire extinguisher','extension cord industrial','electrical cable wire',
            'circuit breaker','water pump submersible','toolbox set',
            'arduino uno','raspberry pi','development board','breadboard',
            'oscilloscope','logic analyzer','function generator',
        ],
    }

    # ── Exact product-type → category path map ────────────────────────────
    _PRODUCT_CATEGORY_MAP = {
        # Phones & Tablets
        'smartphone':           'Phones & Tablets / Phone & Fax / Cell Phones / Smartphones',
        'mobile phone':         'Phones & Tablets / Phone & Fax / Cell Phones / Smartphones',
        'android phone':        'Phones & Tablets / Phone & Fax / Cell Phones / Smartphones',
        'iphone':               'Phones & Tablets / Phone & Fax / Cell Phones / Smartphones',
        'samsung galaxy':       'Phones & Tablets / Phone & Fax / Cell Phones / Smartphones',
        'galaxy s':             'Phones & Tablets / Phone & Fax / Cell Phones / Smartphones',
        'galaxy a':             'Phones & Tablets / Phone & Fax / Cell Phones / Smartphones',
        'infinix':              'Phones & Tablets / Phone & Fax / Cell Phones / Smartphones',
        'tecno camon':          'Phones & Tablets / Phone & Fax / Cell Phones / Smartphones',
        'itel':                 'Phones & Tablets / Phone & Fax / Cell Phones / Smartphones',
        'redmi':                'Phones & Tablets / Phone & Fax / Cell Phones / Smartphones',
        'oppo':                 'Phones & Tablets / Phone & Fax / Cell Phones / Smartphones',
        'vivo':                 'Phones & Tablets / Phone & Fax / Cell Phones / Smartphones',
        'realme':               'Phones & Tablets / Phone & Fax / Cell Phones / Smartphones',
        'oneplus':              'Phones & Tablets / Phone & Fax / Cell Phones / Smartphones',
        'huawei':               'Phones & Tablets / Phone & Fax / Cell Phones / Smartphones',
        'nokia':                'Phones & Tablets / Phone & Fax / Cell Phones / Smartphones',
        'feature phone':        'Phones & Tablets / Phone & Fax / Cell Phones / Basic Phones',
        'tablet':               'Phones & Tablets / Tablets',
        'ipad':                 'Phones & Tablets / Tablets',
        'power bank':           'Phones & Tablets / Accessories / Battery Packs',
        'powerbank':            'Phones & Tablets / Accessories / Battery Packs',
        'screen protector':     'Phones & Tablets / Accessories / Screen Protectors',
        'tempered glass':       'Phones & Tablets / Accessories / Screen Protectors',
        'phone case':           'Phones & Tablets / Accessories / Cases & Sleeves',
        'phone cover':          'Phones & Tablets / Accessories / Cases & Sleeves',
        'phone charger':        'Phones & Tablets / Accessories / Chargers',
        'wireless charger':     'Phones & Tablets / Accessories / Chargers',
        'phone holder':         'Phones & Tablets / Accessories / Holder',
        'selfie stick':         'Phones & Tablets / Accessories / Selfie Sticks',
        'bluetooth headset':    'Phones & Tablets / Accessories / Bluetooth Headsets',
        'earphone':             'Phones & Tablets / Accessories / Headsets',
        'earbuds':              'Phones & Tablets / Accessories / Headsets',
        'sim card':             'Phones & Tablets / Accessories / SIM Cards',
        # Computing
        'laptop':               'Computing / Computers & Accessories / Computers & Tablets / Laptops / Traditional Laptops',
        'notebook':             'Computing / Computers & Accessories / Computers & Tablets / Laptops / Traditional Laptops',
        'gaming laptop':        'Computing / Computers & Accessories / Computers & Tablets / Laptops / Gaming Laptops',
        'desktop computer':     'Computing / Computers & Accessories / Computers & Tablets / Desktop Computers',
        'computer monitor':     'Computing / Computers & Accessories / Monitors',
        'keyboard':             'Computing / Computers & Accessories / Computer Peripherals / Keyboards',
        'computer mouse':       'Computing / Computers & Accessories / Computer Peripherals / Mice',
        'wireless mouse':       'Computing / Computers & Accessories / Computer Peripherals / Mice',
        'laptop bag':           'Computing / Computers & Accessories / Laptop Accessories / Laptop Bags & Cases',
        'laptop stand':         'Computing / Computers & Accessories / Laptop Accessories / Stands',
        'cooling pad':          'Computing / Computers & Accessories / Laptop Accessories / Cooling Pads & External Fans',
        'external hard drive':  'Computing / Computers & Accessories / Data Storage / External Hard Drives',
        'ssd':                  'Computing / Computers & Accessories / Data Storage / Solid State Drives',
        'usb drive':            'Computing / Computers & Accessories / Data Storage / USB Flash Drives',
        'flash drive':          'Computing / Computers & Accessories / Data Storage / USB Flash Drives',
        'pendrive':             'Computing / Computers & Accessories / Data Storage / USB Flash Drives',
        'memory card':          'Computing / Computers & Accessories / Data Storage / Memory Cards',
        'sd card':              'Computing / Computers & Accessories / Data Storage / Memory Cards',
        'wifi router':          'Computing / Computers & Accessories / Networking Products / Routers',
        'wireless router':      'Computing / Computers & Accessories / Networking Products / Routers',
        'ram':                  'Computing / Computers & Accessories / Computer Components / RAM',
        'usb hub':              'Computing / Computers & Accessories / Computer Peripherals / USB Hubs',
        'hdmi cable':           'Computing / Computers & Accessories / Computer Accessories / Cables & Adapters',
        'webcam':               'Computing / Computers & Accessories / Computer Peripherals / Webcams',
        'printer':              'Computing / Printers & Ink / Printers',
        # Electronics
        'smart tv':             'Electronics / Television & Video / Televisions / Smart TVs',
        'led tv':               'Electronics / Television & Video / Televisions / LED & LCD TVs',
        'oled tv':              'Electronics / Television & Video / Televisions / OLED TVs',
        '4k tv':                'Electronics / Television & Video / Televisions / Smart TVs',
        'flat screen':          'Electronics / Television & Video / Televisions / LED & LCD TVs',
        'television':           'Electronics / Television & Video / Televisions / LED & LCD TVs',
        'soundbar':             'Electronics / Television & Video / AV Receivers & Amplifiers / AV Receivers',
        'sound bar':            'Electronics / Television & Video / AV Receivers & Amplifiers / AV Receivers',
        'home theater':         'Electronics / Television & Video / AV Receivers & Amplifiers / AV Receivers',
        'bluetooth speaker':    'Electronics / Portable Audio & Video / Portable Bluetooth Speakers',
        'portable speaker':     'Electronics / Portable Audio & Video / Portable Bluetooth Speakers',
        'wireless headphones':  'Electronics / Portable Audio & Video / Headphones',
        'headphones':           'Electronics / Portable Audio & Video / Headphones',
        'noise cancelling':     'Electronics / Portable Audio & Video / Headphones',
        'tws earbuds':          'Electronics / Portable Audio & Video / Earbuds',
        'security camera':      'Electronics / Camera & Photo / Security & Surveillance / Surveillance Video Equipment',
        'cctv':                 'Electronics / Camera & Photo / Security & Surveillance / Surveillance Video Equipment',
        'ip camera':            'Electronics / Camera & Photo / Security & Surveillance / Surveillance Video Equipment',
        'dash cam':             'Electronics / Camera & Photo / Video Surveillance / Vehicle Cameras',
        'digital camera':       'Electronics / Camera & Photo / Digital Cameras',
        'dslr camera':          'Electronics / Camera & Photo / Digital Cameras / DSLR',
        'action camera':        'Electronics / Camera & Photo / Action Cameras',
        'drone':                'Electronics / Camera & Photo / Drones',
        'projector':            'Electronics / Television & Video / Projectors',
        'android tv box':       'Electronics / Television & Video / Streaming Media Players',
        'streaming device':     'Electronics / Television & Video / Streaming Media Players',
        'dvd player':           'Electronics / Television & Video / DVD Players & Recorders / DVD Players',
        'generator':            'Electronics / Power Accessories / Generators',
        'inverter':             'Electronics / Power Accessories / Inverters',
        'solar panel':          'Electronics / Power Accessories / Solar Panels',
        # Health & Beauty
        'hair dryer':           'Health & Beauty / Beauty & Personal Care / Hair Care / Styling Tools & Appliances / Hair Dryers & Accessories / Hair Dryers',
        'blow dryer':           'Health & Beauty / Beauty & Personal Care / Hair Care / Styling Tools & Appliances / Hair Dryers & Accessories / Hair Dryers',
        'hair straightener':    'Health & Beauty / Beauty & Personal Care / Hair Care / Styling Tools & Appliances / Hair Straighteners',
        'flat iron hair':       'Health & Beauty / Beauty & Personal Care / Hair Care / Styling Tools & Appliances / Hair Straighteners',
        'curling iron':         'Health & Beauty / Beauty & Personal Care / Hair Care / Styling Tools & Appliances / Curling Irons',
        'hair clipper':         'Health & Beauty / Beauty & Personal Care / Shave & Hair Removal / Mens / Electric Shavers & Clippers',
        'electric shaver':      'Health & Beauty / Beauty & Personal Care / Shave & Hair Removal / Mens / Electric Shavers & Clippers',
        'electric trimmer':     'Health & Beauty / Beauty & Personal Care / Shave & Hair Removal / Mens / Trimmers',
        # ── Shaver / razor / grooming — high-specificity keys added to beat
        # single-word 'rechargeable' / 'rotary' routing to wrong domains ──────
        'beard shaver':         'Health & Beauty / Beauty & Personal Care / Shave & Hair Removal / Mens / Electric Shavers & Clippers',
        'beard razor':          'Health & Beauty / Beauty & Personal Care / Shave & Hair Removal / Mens / Electric Shavers & Clippers',
        'beard trimmer':        'Health & Beauty / Beauty & Personal Care / Shave & Hair Removal / Mens / Trimmers',
        'rotary shaver':        'Health & Beauty / Beauty & Personal Care / Shave & Hair Removal / Mens / Electric Shavers & Clippers',
        'rotary head shaver':   'Health & Beauty / Beauty & Personal Care / Shave & Hair Removal / Mens / Electric Shavers & Clippers',
        'rotary head razor':    'Health & Beauty / Beauty & Personal Care / Shave & Hair Removal / Mens / Electric Shavers & Clippers',
        'rechargeable shaver':  'Health & Beauty / Beauty & Personal Care / Shave & Hair Removal / Mens / Electric Shavers & Clippers',
        'rechargeable razor':   'Health & Beauty / Beauty & Personal Care / Shave & Hair Removal / Mens / Electric Shavers & Clippers',
        'rechargeable trimmer': 'Health & Beauty / Beauty & Personal Care / Shave & Hair Removal / Mens / Trimmers',
        'cordless trimmer':     'Health & Beauty / Beauty & Personal Care / Shave & Hair Removal / Mens / Trimmers',
        'cordless shaver':      'Health & Beauty / Beauty & Personal Care / Shave & Hair Removal / Mens / Electric Shavers & Clippers',
        'electric razor':       'Health & Beauty / Beauty & Personal Care / Shave & Hair Removal / Mens / Electric Shavers & Clippers',
        'shaving machine':      'Health & Beauty / Beauty & Personal Care / Shave & Hair Removal / Mens / Electric Shavers & Clippers',
        'grooming tool':        'Health & Beauty / Beauty & Personal Care / Shave & Hair Removal / Mens / Electric Shavers & Clippers',
        'grooming machine':     'Health & Beauty / Beauty & Personal Care / Shave & Hair Removal / Mens / Electric Shavers & Clippers',
        # ── Space heaters — prevent routing to Automobile/Heater ──────────────
        'space heater':         'Home & Office / Home & Kitchen / Heating, Cooling & Air Quality / Space Heaters',
        'electric heater':      'Home & Office / Home & Kitchen / Heating, Cooling & Air Quality / Space Heaters',
        'ceramic heater':       'Home & Office / Home & Kitchen / Heating, Cooling & Air Quality / Space Heaters',
        'quartz heater':        'Home & Office / Home & Kitchen / Heating, Cooling & Air Quality / Space Heaters',
        'ptc heater':           'Home & Office / Home & Kitchen / Heating, Cooling & Air Quality / Space Heaters',
        'heater fan':           'Home & Office / Home & Kitchen / Heating, Cooling & Air Quality / Space Heaters',
        # ── Herbal supplements / powders in Wholesale — route to H&B ─────────
        'herbal supplement':    'Health & Beauty / Vitamins & Dietary Supplements / Herbal Supplements',
        'herbal powder':        'Health & Beauty / Vitamins & Dietary Supplements / Herbal Supplements',
        'natural supplement':   'Health & Beauty / Vitamins & Dietary Supplements / Herbal Supplements',
        'detox powder':         'Health & Beauty / Vitamins & Dietary Supplements / Herbal Supplements',
        'weight management powder': 'Health & Beauty / Sports Nutrition / Weight Management',
        'libido supplement':    'Health & Beauty / Vitamins & Dietary Supplements / Herbal Supplements',
        'mukombero':            'Health & Beauty / Vitamins & Dietary Supplements / Herbal Supplements',
        'moringa powder':       'Health & Beauty / Vitamins & Dietary Supplements / Herbal Supplements',
        'green tea powder':     'Health & Beauty / Vitamins & Dietary Supplements / Herbal Supplements',
        'organic matcha':       'Health & Beauty / Vitamins & Dietary Supplements / Herbal Supplements',
        'matcha tea':           'Health & Beauty / Vitamins & Dietary Supplements / Herbal Supplements',
        'soursop powder':       'Health & Beauty / Vitamins & Dietary Supplements / Herbal Supplements',
        'garcinia':             'Health & Beauty / Sports Nutrition / Weight Management',
        'shilajit':             'Health & Beauty / Vitamins & Dietary Supplements / Herbal Supplements',
        'mumiyo':               'Health & Beauty / Vitamins & Dietary Supplements / Herbal Supplements',
        'chasteberry':          'Health & Beauty / Vitamins & Dietary Supplements / Herbal Supplements',
        'matcha powder':        'Health & Beauty / Vitamins & Dietary Supplements / Herbal Supplements',
        # ── Smart watches — keep in Phones & Tablets ─────────────────────────
        'smart watch':          'Phones & Tablets / Wearable Technology / Smart Watches',
        'smartwatch':           'Phones & Tablets / Wearable Technology / Smart Watches',
        'fitness tracker':      'Phones & Tablets / Wearable Technology / Smart Watches',
        # ── Bluetooth audio earmuffs — Electronics not Fashion ───────────────
        'bluetooth earmuff':    'Electronics / Portable Audio & Video / Headphones',
        'wireless earmuff':     'Electronics / Portable Audio & Video / Headphones',
        'earmuff headphone':    'Electronics / Portable Audio & Video / Headphones',
        'ear warmer headphone': 'Electronics / Portable Audio & Video / Headphones',
        # ── Walking aids — Health & Beauty not Fashion ───────────────────────
        'walking cane':         'Health & Beauty / Medical Supplies & Equipment / Mobility & Daily Living Aids / Walking Canes',
        'walking stick':        'Health & Beauty / Medical Supplies & Equipment / Mobility & Daily Living Aids / Walking Canes',
        'adjustable walking':   'Health & Beauty / Medical Supplies & Equipment / Mobility & Daily Living Aids / Walking Canes',
        'foldable cane':        'Health & Beauty / Medical Supplies & Equipment / Mobility & Daily Living Aids / Walking Canes',
        # ── Food dehydrator — prevent routing to Freezers / Grocery ─────────
        'food dehydrator':      'Home & Office / Home & Kitchen / Kitchen & Dining / Kitchen Appliances / Food Dehydrators',
        'fruit dehydrator':     'Home & Office / Home & Kitchen / Kitchen & Dining / Kitchen Appliances / Food Dehydrators',
        'vegetable dehydrator': 'Home & Office / Home & Kitchen / Kitchen & Dining / Kitchen Appliances / Food Dehydrators',
        'food drying machine':  'Home & Office / Home & Kitchen / Kitchen & Dining / Kitchen Appliances / Food Dehydrators',
        # ── Spray gun / power tools — prevent routing to Grocery ─────────────
        'spray gun':            'Home & Office / Tools & Home Improvement / Power & Hand Tools / Power Tools / Spray Guns',
        'electric spray gun':   'Home & Office / Tools & Home Improvement / Power & Hand Tools / Power Tools / Spray Guns',
        'cordless spray gun':   'Home & Office / Tools & Home Improvement / Power & Hand Tools / Power Tools / Spray Guns',
        # ── Perfume / eau de parfum — Health & Beauty not Computing ──────────
        'eau de parfum':        'Health & Beauty / Beauty & Personal Care / Fragrance / Womens Perfume',
        'eau de toilette':      'Health & Beauty / Beauty & Personal Care / Fragrance / Womens Perfume',
        'edp perfume':          'Health & Beauty / Beauty & Personal Care / Fragrance / Womens Perfume',
        'hair extension':       'Health & Beauty / Beauty & Personal Care / Hair Care / Extensions, Wigs & Accessories / Hair Extensions',
        'hair weave':           'Health & Beauty / Beauty & Personal Care / Hair Care / Extensions, Wigs & Accessories / Hair Extensions',
        'lace front wig':       'Health & Beauty / Beauty & Personal Care / Hair Care / Extensions, Wigs & Accessories / Wigs',
        'human hair wig':       'Health & Beauty / Beauty & Personal Care / Hair Care / Extensions, Wigs & Accessories / Wigs',
        'wig':                  'Health & Beauty / Beauty & Personal Care / Hair Care / Extensions, Wigs & Accessories / Wigs',
        'foundation':           'Health & Beauty / Beauty & Personal Care / Makeup / Face / Foundation',
        'bb cream':             'Health & Beauty / Beauty & Personal Care / Makeup / Face / Foundation',
        'concealer':            'Health & Beauty / Beauty & Personal Care / Makeup / Face / Concealer & Neutralizer',
        'lipstick':             'Health & Beauty / Beauty & Personal Care / Makeup / Lips / Lipstick',
        'lip gloss':            'Health & Beauty / Beauty & Personal Care / Makeup / Lips / Lip Gloss',
        'eyeshadow':            'Health & Beauty / Beauty & Personal Care / Makeup / Eyes / Eye Shadow',
        'mascara':              'Health & Beauty / Beauty & Personal Care / Makeup / Eyes / Mascara',
        'eyeliner':             'Health & Beauty / Beauty & Personal Care / Makeup / Eyes / Eyeliner',
        'face serum':           'Health & Beauty / Beauty & Personal Care / Skin Care / Serums & Essences',
        'face cream':           'Health & Beauty / Beauty & Personal Care / Skin Care / Face Moisturizers',
        'moisturizer':          'Health & Beauty / Beauty & Personal Care / Skin Care / Face Moisturizers',
        'sunscreen':            'Health & Beauty / Beauty & Personal Care / Skin Care / Sunscreen',
        'body lotion':          'Health & Beauty / Beauty & Personal Care / Skin Care / Body Lotions',
        'body cream':           'Health & Beauty / Beauty & Personal Care / Skin Care / Body Lotions',
        'face wash':            'Health & Beauty / Beauty & Personal Care / Skin Care / Face Cleansers',
        'shampoo':              'Health & Beauty / Beauty & Personal Care / Hair Care / Shampoo',
        'conditioner':          'Health & Beauty / Beauty & Personal Care / Hair Care / Conditioners',
        'hair oil':             'Health & Beauty / Beauty & Personal Care / Hair Care / Hair & Scalp Treatments',
        'perfume':              'Health & Beauty / Beauty & Personal Care / Fragrance / Womens Perfume',
        'cologne':              'Health & Beauty / Beauty & Personal Care / Fragrance / Mens Cologne',
        'deodorant':            'Health & Beauty / Beauty & Personal Care / Deodorants & Antiperspirants',
        'blood pressure monitor': 'Health & Beauty / Medical Supplies & Equipment / Health Monitors / Blood Pressure Monitors',
        'glucometer':           'Health & Beauty / Medical Supplies & Equipment / Health Monitors / Blood Glucose Monitors',
        'glucose meter':        'Health & Beauty / Medical Supplies & Equipment / Health Monitors / Blood Glucose Monitors',
        'pulse oximeter':       'Health & Beauty / Medical Supplies & Equipment / Health Monitors / Pulse Oximeters',
        'thermometer':          'Health & Beauty / Medical Supplies & Equipment / Health Monitors / Thermometers',
        'sanitary pad':         'Health & Beauty / Health Care / Feminine Care / Menstrual Pads',
        'menstrual pad':        'Health & Beauty / Health Care / Feminine Care / Menstrual Pads',
        'massage gun':          'Health & Beauty / Health Care / Massage & Relaxation / Massagers',
        'whey protein':         'Health & Beauty / Sports Nutrition / Protein Supplements',
        'protein powder':       'Health & Beauty / Sports Nutrition / Protein Supplements',
        'fish oil':             'Health & Beauty / Vitamins & Dietary Supplements / Fish Oil & Omega Fatty Acids',
        'collagen':             'Health & Beauty / Vitamins & Dietary Supplements / Collagen Supplements',
        'vitamin':              'Health & Beauty / Vitamins & Dietary Supplements / Multivitamins',
        # Home & Office
        'sofa':                 'Home & Office / Home & Kitchen / Furniture / Living Room Furniture / Sofas & Couches',
        'couch':                'Home & Office / Home & Kitchen / Furniture / Living Room Furniture / Sofas & Couches',
        'bed frame':            'Home & Office / Home & Kitchen / Furniture / Bedroom Furniture / Beds & Bed Frames',
        'mattress':             'Home & Office / Home & Kitchen / Furniture / Bedroom Furniture / Mattresses',
        'wardrobe':             'Home & Office / Home & Kitchen / Furniture / Bedroom Furniture / Wardrobes & Armoires',
        'dressing table':       'Home & Office / Home & Kitchen / Furniture / Bedroom Furniture / Vanities & Dressing Tables',
        'dining table':         'Home & Office / Home & Kitchen / Furniture / Kitchen & Dining Room Furniture / Dining Tables',
        'office desk':          'Home & Office / Home & Kitchen / Furniture / Home Office Furniture / Desks',
        'standing desk':        'Home & Office / Home & Kitchen / Furniture / Home Office Furniture / Desks',
        'office chair':         'Home & Office / Home & Kitchen / Furniture / Home Office Furniture / Home Office Desk Chairs',
        'bookshelf':            'Home & Office / Home & Kitchen / Furniture / Living Room Furniture / Bookcases & Shelving',
        'tv stand':             'Home & Office / Home & Kitchen / Furniture / Living Room Furniture / TV Stands & Entertainment Centers',
        'wall clock':           'Home & Office / Home & Kitchen / Home Décor / Clocks',
        'curtain':              'Home & Office / Home & Kitchen / Home Décor / Window Treatments / Curtains',
        'ceiling fan':          'Home & Office / Home & Kitchen / Heating, Cooling & Air Quality / Fans / Ceiling Fans',
        'standing fan':         'Home & Office / Home & Kitchen / Heating, Cooling & Air Quality / Fans / Tower & Pedestal Fans',
        'air conditioner':      'Home & Office / Home & Kitchen / Heating, Cooling & Air Quality / Air Conditioners',
        'refrigerator':         'Home & Office / Home & Kitchen / Appliances / Refrigerators',
        'fridge':               'Home & Office / Home & Kitchen / Appliances / Refrigerators',
        'washing machine':      'Home & Office / Home & Kitchen / Appliances / Washing Machines & Dryers',
        'microwave':            'Home & Office / Home & Kitchen / Appliances / Microwaves',
        'blender':              'Home & Office / Home & Kitchen / Kitchen & Dining / Kitchen Appliances / Blenders',
        'toaster':              'Home & Office / Home & Kitchen / Kitchen & Dining / Kitchen Appliances / Toasters & Ovens',
        'electric kettle':      'Home & Office / Home & Kitchen / Kitchen & Dining / Kitchen Appliances / Electric Kettles',
        'cooking pot':          'Home & Office / Home & Kitchen / Kitchen & Dining / Cookware / Pots & Pans',
        'frying pan':           'Home & Office / Home & Kitchen / Kitchen & Dining / Cookware / Frying Pans & Skillets',
        'clothes iron':         'Home & Office / Home & Kitchen / Laundry / Irons & Steamers / Clothing Irons',
        'steam iron':           'Home & Office / Home & Kitchen / Laundry / Irons & Steamers / Clothing Irons',
        'vacuum cleaner':       'Home & Office / Home & Kitchen / Appliances / Vacuum Cleaners',
        'paper shredder':       'Home & Office / Office Products / Office Machines / Paper Shredders',
        'whiteboard':           'Home & Office / Office Products / Office & School Supplies / Presentation & Bulletin Boards',
        'bed sheet':            'Home & Office / Home & Kitchen / Bedding / Bed Sheets & Pillowcases',
        'pillow':               'Home & Office / Home & Kitchen / Bedding / Pillows',
        'duvet':                'Home & Office / Home & Kitchen / Bedding / Comforters & Duvets',
        'storage box':          'Home & Office / Home & Kitchen / Storage & Organization / Storage Boxes & Bins',
        'sewing machine':       'Home & Office / Home & Kitchen / Appliances / Sewing Machines & Accessories',
        'instant coffee':       'Home & Office / Home & Kitchen / Kitchen & Dining / Coffee, Tea & Espresso / Coffee Makers / Coffee Machines',
        'nescafe':              'Home & Office / Home & Kitchen / Kitchen & Dining / Coffee, Tea & Espresso / Coffee Makers / Coffee Machines',
        'cordless drill':       'Home & Office / Tools & Home Improvement / Power & Hand Tools / Power Tools / Drills / Drill Drivers',
        'rice bag':             'Home & Office / Home & Kitchen / Kitchen & Dining / Small Appliances / Rice Cookers',
        'rice grain':           'Home & Office / Home & Kitchen / Kitchen & Dining / Small Appliances / Rice Cookers',
        'long grain rice':      'Home & Office / Home & Kitchen / Kitchen & Dining / Small Appliances / Rice Cookers',
        'parboiled rice':       'Home & Office / Home & Kitchen / Kitchen & Dining / Small Appliances / Rice Cookers',
        'cooking oil':          'Home & Office / Home & Kitchen / Kitchen & Dining / Small Appliances / Coffee, Tea & Espresso Appliances',
        # Automobile
        'car seat cover':       'Automobile / Interior Accessories / Covers / Seat Covers',
        'car mat':              'Automobile / Interior Accessories / Floor Mats & Cargo Liners',
        'car floor mat':        'Automobile / Interior Accessories / Floor Mats & Cargo Liners',
        'car air freshener':    'Automobile / Interior Accessories / Air Fresheners',
        'car phone mount':      'Automobile / Car Electronics & Accessories / GPS & Navigation / GPS Holders & Mounts',
        'dashboard camera':     'Automobile / Car Electronics & Accessories / Car Electronics / Dash Cameras',
        'car stereo':           'Automobile / Car Electronics & Accessories / Car Electronics / Car Audio / Car Stereo Receivers',
        'car radio':            'Automobile / Car Electronics & Accessories / Car Electronics / Car Audio / Car Stereo Receivers',
        'car alarm':            'Automobile / Car Electronics & Accessories / Car Electronics / Car Security',
        'engine oil':           'Automobile / Oils & Fluids / Oils / Motor Oils',
        'motor oil':            'Automobile / Oils & Fluids / Oils / Motor Oils',
        'gear oil':             'Automobile / Oils & Fluids / Transmission Fluid',
        'brake pad':            'Automobile / Replacement Parts / Brake Parts / Brake Pads & Shoes',
        'brake disc':           'Automobile / Replacement Parts / Brake Parts / Rotors',
        'car battery':          'Automobile / Replacement Parts / Electrical / Batteries',
        'spark plug':           'Automobile / Replacement Parts / Ignition Parts / Spark Plugs',
        'car headlight':        'Automobile / Replacement Parts / Lighting / Headlights',
        'tyre':                 'Automobile / Tires & Wheels / Car & Truck Tires',
        'car tire':             'Automobile / Tires & Wheels / Car & Truck Tires',
        'wheel rim':            'Automobile / Tires & Wheels / Wheels & Hubcaps',
        'tire inflator':        'Automobile / Tools & Equipment / Tire Inflators & Compressors',
        'jump starter':         'Automobile / Tools & Equipment / Jump Starters',
        'car cover':            'Automobile / Exterior Accessories / Car Covers',
        'windshield wiper':     'Automobile / Replacement Parts / Exterior Parts / Windshield Wipers',
        'car wax':              'Automobile / Car Care / Exterior Care / Car Polishes & Waxes',
        'car wash':             'Automobile / Car Care / Exterior Care / Car Wash Equipment',
        'motorcycle helmet':    'Automobile / Motorcycle & Powersports / Protective Gear / Helmets',
        # Baby Products
        'baby formula':         'Baby Products / Feeding / Formula',
        'infant formula':       'Baby Products / Feeding / Formula',
        'baby food':            'Baby Products / Feeding / Baby Food',
        'diaper':               'Baby Products / Diapering / Diapers',
        'baby nappy':           'Baby Products / Diapering / Diapers',
        'baby wipes':           'Baby Products / Diapering / Baby Wipes',
        'baby lotion':          'Baby Products / Baby Care / Lotions & Creams',
        'baby oil':             'Baby Products / Baby Care / Oils',
        'baby powder':          'Baby Products / Baby Care / Powders',
        'baby shampoo':         'Baby Products / Baby Care / Shampoo & Wash',
        'baby bottle':          'Baby Products / Feeding / Bottles',
        'pacifier':             'Baby Products / Feeding / Pacifiers & Teethers',
        'baby monitor':         'Baby Products / Baby Safety / Baby Monitors',
        'baby stroller':        'Baby Products / Strollers & Accessories / Strollers',
        'pram':                 'Baby Products / Strollers & Accessories / Strollers',
        'baby cot':             'Baby Products / Furniture / Cribs & Toddler Beds',
        'baby crib':            'Baby Products / Furniture / Cribs & Toddler Beds',
        'high chair':           'Baby Products / Feeding / Highchairs',
        'baby walker':          'Baby Products / Activity & Entertainment / Baby Walkers',
        'baby bouncer':         'Baby Products / Activity & Entertainment / Bouncers & Vibrating Seats',
        'baby swing':           'Baby Products / Activity & Entertainment / Swings',
        'baby blanket':         'Baby Products / Nursery / Blankets & Swaddling',
        'baby romper':          'Baby Products / Apparel & Accessories / Bodysuits & One-Pieces',
        # Fashion — Women's (longest keys first so they win over shorter ones)
        # Nightwear / Sleepwear
        'ladies quality nightwear top and shorts': "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Sets",
        'ladies quality nightwear top and trouser': "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Sets",
        'ladies quality stretchable camisole': "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Lingerie / Camisoles & Tanks",
        'ladies casual top and pants set': "Fashion / Women's Fashion / Clothing / Pants",
        'ladies sexy top and short':    "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Sets",
        'ladies top and trouser':       "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Sets",
        'ladies top and short':         "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Sets",
        'ladies night gown':            "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Nightgowns & Sleepshirts",
        'ladies sexy gown':             "Fashion / Women's Fashion / Clothing / Dresses",
        'ladies beautiful gown':        "Fashion / Women's Fashion / Clothing / Dresses",
        'ladies characters gown':       "Fashion / Women's Fashion / Clothing / Dresses",
        'ladies nightwear':             "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Sets",
        'ladies sleepwear':             "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Sets",
        'ladies night wear':            "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Sets",
        'nightwear for ladies':         "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Sets",
        'sleepwear for ladies':         "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Sets",
        'nightwear for women':          "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Sets",
        'night wear for ladies':        "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Sets",
        'nightgown for ladies':         "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Nightgowns & Sleepshirts",
        'ladies nightgown':             "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Nightgowns & Sleepshirts",
        'women nightwear':              "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Sets",
        'women sleepwear':              "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Sets",
        'women night gown':             "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Nightgowns & Sleepshirts",
        'sexy nightwear':               "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Sets",
        'beautiful nightwear':          "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Sets",
        'classy nightwear':             "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Sets",
        'lovely nightwear':             "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Sets",
        'elegant nightwear':            "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Sets",
        # Lingerie / Panties
        'ladies condom panties':        "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Lingerie / Panties",
        'ladies nika panties':          "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Lingerie / Panties",
        'ladies menstrual panties':     "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Lingerie / Panties",
        'ladies net lace panties':      "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Lingerie / Panties",
        'ladies cotton panties':        "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Lingerie / Panties",
        'ladies panties':               "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Lingerie / Panties",
        'ladies underwear':             "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Lingerie / Panties",
        'panties for ladies':           "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Lingerie / Panties",
        'ladies v-shape panties':       "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Lingerie / Panties",
        'ladies sexy pant':             "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Lingerie / Panties",
        'ladies seamless pant':         "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Lingerie / Panties",
        'ladies cotton pant':           "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Lingerie / Panties",
        'sexy ladies pant':             "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Lingerie / Panties",
        'female pant':                  "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Lingerie / Panties",
        'sexy panties':                 "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Lingerie / Panties",
        'g-string panties':             "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Lingerie / Panties",
        'g-string ladies':              "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Lingerie / Panties",
        'gstring ladies':               "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Lingerie / Panties",
        'sexy g-string':                "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Lingerie / Panties",
        'cotton sexy gstring':          "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Lingerie / Panties",
        'boyshorts':                    "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Lingerie / Panties",
        'women sexy boyshorts':         "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Lingerie / Panties",
        'ladies lingerie':              "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Lingerie",
        'sexy ladies lingerie':         "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Lingerie",
        'sexy lingerie':                "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Lingerie",
        # Camisole / Singlet / Tops
        'ladies stretchable camisole':  "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Lingerie / Camisoles & Tanks",
        'ladies sexy stretchable camisole': "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Lingerie / Camisoles & Tanks",
        'ladies camisole':              "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Lingerie / Camisoles & Tanks",
        'ladies singlet top':           "Fashion / Women's Fashion / Clothing / Tops & Tees",
        'ladies stretchable tops':      "Fashion / Women's Fashion / Clothing / Tops & Tees",
        'ladies quality tops':          "Fashion / Women's Fashion / Clothing / Tops & Tees",
        'ladies stretchable top':       "Fashion / Women's Fashion / Clothing / Tops & Tees",
        'ladies sexy top':              "Fashion / Women's Fashion / Clothing / Tops & Tees",
        'ladies casual top':            "Fashion / Women's Fashion / Clothing / Tops & Tees",
        'ladies quality top':           "Fashion / Women's Fashion / Clothing / Tops & Tees",
        # Bra
        'breathable bra for ladies':    "Fashion / Women's Fashion / Underwear & Sleepwear / Bra",
        'bra for ladies':               "Fashion / Women's Fashion / Underwear & Sleepwear / Bra",
        'ladies push up multicolored bra': "Fashion / Women's Fashion / Underwear & Sleepwear / Bra",
        'ladies push up bra':           "Fashion / Women's Fashion / Underwear & Sleepwear / Bra",
        'ladies push up':               "Fashion / Women's Fashion / Underwear & Sleepwear / Bra",
        'ladies bra':                   "Fashion / Women's Fashion / Underwear & Sleepwear / Bra",
        # Leggings / Tights
        'bumper tight for ladies':      "Fashion / Women's Fashion / Clothing / Leggings",
        'bumper tight':                 "Fashion / Women's Fashion / Clothing / Leggings",
        'biker shorts':                 "Fashion / Women's Fashion / Clothing / Leggings",
        'biker short':                  "Fashion / Women's Fashion / Clothing / Leggings",
        'ladies tight':                 "Fashion / Women's Fashion / Clothing / Leggings",
        'ladies sexy tight':            "Fashion / Women's Fashion / Clothing / Leggings",
        'ladies legging':               "Fashion / Women's Fashion / Clothing / Leggings",
        'sexy tight ladies':            "Fashion / Women's Fashion / Clothing / Leggings",
        'sexy yoga pants':              "Fashion / Women's Fashion / Clothing / Leggings",
        'female tight':                 "Fashion / Women's Fashion / Clothing / Leggings",
        # Jeans
        'baggy jeans for ladies':       "Fashion / Women's Fashion / Clothing / Jeans",
        'cargo jean for ladies':        "Fashion / Women's Fashion / Clothing / Jeans",
        'boyfriend cargo jean':         "Fashion / Women's Fashion / Clothing / Jeans",
        'ladies jeans':                 "Fashion / Women's Fashion / Clothing / Jeans",
        'ladies jean':                  "Fashion / Women's Fashion / Clothing / Jeans",
        # Dress / Gown / Kaftan
        'ladies kaftan':                "Fashion / Women's Fashion / Clothing / Dresses",
        'ladies abaya':                 "Fashion / Women's Fashion / Clothing / Dresses",
        'ladies kimono':                "Fashion / Women's Fashion / Clothing / Dresses",
        'ladies dress':                 "Fashion / Women's Fashion / Clothing / Dresses",
        'ladies gown':                  "Fashion / Women's Fashion / Clothing / Dresses",
        'women kaftan':                 "Fashion / Women's Fashion / Clothing / Dresses",
        'bubu kaftan':                  "Fashion / Women's Fashion / Clothing / Dresses",
        'african dresses for women':    "Fashion / Women's Fashion / Clothing / Dresses",
        'maxi dress':                   "Fashion / Women's Fashion / Clothing / Dresses",
        'evening gown':                 "Fashion / Women's Fashion / Clothing / Dresses",
        # Shorts
        'ladies shorts':                "Fashion / Women's Fashion / Clothing / Shorts",
        # Men's Fashion
        'agbada':               'Fashion / Mens Fashion / Traditional & Cultural Wear / African',
        'senator suit':         'Fashion / Mens Fashion / Traditional & Cultural Wear / African',
        'kaftan':               'Fashion / Mens Fashion / Traditional & Cultural Wear / African',
        'aso oke':              'Fashion / Traditional & Cultural Wear / Women / African',
        'shapewear':            'Fashion / Womens Fashion / Lingerie, Sleep & Lounge / Lingerie / Shapewear',
        'waist trainer':        'Fashion / Womens Fashion / Lingerie, Sleep & Lounge / Lingerie / Shapewear / Waist Cinchers',
        'bra':                  "Fashion / Women's Fashion / Underwear & Sleepwear / Bra",
        'panties':              "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Lingerie / Panties",
        'polo shirt':           'Fashion / Mens Fashion / Clothing / Shirts',
        'boxer shorts':         'Fashion / Mens Fashion / Underwear & Sleepwear / Underwear',
        'boxer brief':          'Fashion / Mens Fashion / Underwear & Sleepwear / Underwear',
        'men boxers':           'Fashion / Mens Fashion / Underwear & Sleepwear / Underwear',
        'school uniform':       'Fashion / Kids Fashion / Uniforms',
        'ladies handbag':       'Fashion / Womens Fashion / Bags / Handbags',
        'ladies heels':         'Fashion / Womens Fashion / Shoes / Pumps & Heels',
        'ladies sandals':       'Fashion / Womens Fashion / Shoes / Sandals',
        'men sneakers':         'Fashion / Mens Fashion / Shoes / Sneakers',
        'men shoes':            'Fashion / Mens Fashion / Shoes',
        'necklace':             'Fashion / Womens Fashion / Jewelry / Necklaces',
        'bracelet':             'Fashion / Womens Fashion / Jewelry / Bracelets',
        'earrings':             'Fashion / Womens Fashion / Jewelry / Earrings',
        # Sporting Goods
        'football':             'Sporting Goods / Team Sports / Soccer / Balls',
        'soccer ball':          'Sporting Goods / Team Sports / Soccer / Balls',
        'basketball':           'Sporting Goods / Team Sports / Basketball / Balls',
        'tennis racket':        'Sporting Goods / Racket Sports / Tennis / Rackets',
        'badminton':            'Sporting Goods / Racket Sports / Badminton / Rackets',
        'treadmill':            'Sporting Goods / Sports & Fitness / Exercise & Fitness / Cardio Training / Treadmills',
        'exercise bike':        'Sporting Goods / Sports & Fitness / Exercise & Fitness / Cardio Training / Exercise Bikes',
        'dumbbell':             'Sporting Goods / Sports & Fitness / Exercise & Fitness / Strength Training / Dumbbells',
        'barbell':              'Sporting Goods / Sports & Fitness / Exercise & Fitness / Strength Training / Barbells',
        'kettlebell':           'Sporting Goods / Sports & Fitness / Exercise & Fitness / Strength Training / Kettlebells',
        'yoga mat':             'Sporting Goods / Sports & Fitness / Exercise & Fitness / Yoga / Yoga Mats',
        'jump rope':            'Sporting Goods / Sports & Fitness / Exercise & Fitness / Cardio Training / Jump Ropes',
        'boxing gloves':        'Sporting Goods / Sports & Fitness / Other Sports / Boxing / Boxing Gloves',
        'fishing rod':          'Sporting Goods / Sports & Fitness / Hunting & Fishing / Fishing / Rods / Spinning Rods',
        'camping tent':         'Sporting Goods / Sports & Fitness / Outdoor Recreation / Camping & Hiking / Tents & Shelters',
        'sleeping bag':         'Sporting Goods / Sports & Fitness / Outdoor Recreation / Camping & Hiking / Sleeping Bags',
        'football boots':       'Sporting Goods / Team Sports / Soccer / Footwear',
        'gym bag':              'Sporting Goods / Sports & Fitness / Exercise & Fitness / Sports Bags',
        'running shoes':        'Sporting Goods / Sports & Fitness / Exercise & Fitness / Footwear',
        'sport shoes':          'Sporting Goods / Sports & Fitness / Exercise & Fitness / Footwear',
        # Toys & Games
        'lego':                 'Toys & Games / Building Toys / Building Sets / LEGO Sets',
        'building blocks':      'Toys & Games / Building Toys / Building Sets',
        'action figure':        'Toys & Games / Dolls & Action Figures / Action Figures',
        'barbie doll':          'Toys & Games / Dolls & Action Figures / Dolls / Fashion Dolls',
        'rc car':               'Toys & Games / Remote Control / Cars',
        'remote control car':   'Toys & Games / Remote Control / Cars',
        'rc remote control':    'Toys & Games / Remote Control / Cars',
        'board game':           'Toys & Games / Games / Board Games',
        'jigsaw puzzle':        'Toys & Games / Puzzles / Jigsaw Puzzles',
        'teddy bear':           'Toys & Games / Stuffed Animals & Plush Toys / Teddy Bears',
        'plush toy':            'Toys & Games / Stuffed Animals & Plush Toys',
        'playdoh':              'Toys & Games / Arts & Crafts / Modeling Dough & Clay',
        # Gaming
        'ps5':                  'Gaming / Playstation / PlayStation 5 / Consoles',
        'playstation 5':        'Gaming / Playstation / PlayStation 5 / Consoles',
        'ps4':                  'Gaming / Playstation / PlayStation 4 / Consoles',
        'xbox series':          'Gaming / Xbox / Xbox Series X|S / Consoles',
        'nintendo switch':      'Gaming / Nintendo / Nintendo Switch / Consoles',
        'gaming controller':    'Gaming / Accessories / Controllers',
        'wireless gamepad':     'Gaming / Accessories / Controllers',
        'gaming headset':       'Gaming / Accessories / Headsets',
        'gaming chair':         'Gaming / Accessories / Gaming Chairs',
        'gaming keyboard':      'Gaming / Accessories / Keyboards',
        'gaming mouse':         'Gaming / Accessories / Mice',
        'gaming monitor':       'Gaming / Accessories / Monitors',
        # Pet Supplies
        'dog food':             'Pet Supplies / Dogs / Food / Dry',
        'cat food':             'Pet Supplies / Cats / Food / Dry',
        'fish food':            'Pet Supplies / Fish & Aquatic Pets / Food',
        'bird food':            'Pet Supplies / Birds / Food',
        'dog treat':            'Pet Supplies / Dogs / Treats',
        'cat treat':            'Pet Supplies / Cats / Treats',
        'dog collar':           'Pet Supplies / Dogs / Collars & Leads / Collars',
        'pet carrier':          'Pet Supplies / Dogs / Carriers & Travel Products / Carrier Bags',
        'bird cage':            'Pet Supplies / Birds / Cages & Accessories / Cages',
        'fish tank':            'Pet Supplies / Fish & Aquatic Pets / Aquariums & Accessories / Aquariums',
        'aquarium':             'Pet Supplies / Fish & Aquatic Pets / Aquariums & Accessories / Aquariums',
        'pet bed':              'Pet Supplies / Dogs / Beds & Furniture / Beds',
        'litter box':           'Pet Supplies / Cats / Litter & Housebreaking / Litter Boxes',
        'cat litter':           'Pet Supplies / Cats / Litter & Housebreaking / Cat Litter',
        'pet shampoo':          'Pet Supplies / Dogs / Grooming / Shampoos & Conditioners',
        # Garden & Outdoors
        'garden hose':          'Garden & Outdoors / Gardening & Lawn Care / Watering Equipment / Garden Hoses',
        'lawn mower':           'Garden & Outdoors / Gardening & Lawn Care / Lawn Mowers & Tractors / Push Mowers',
        'plant pot':            'Garden & Outdoors / Gardening & Lawn Care / Pots, Planters & Accessories / Pots & Planters',
        'flower pot':           'Garden & Outdoors / Gardening & Lawn Care / Pots, Planters & Accessories / Pots & Planters',
        'fertilizer':           'Garden & Outdoors / Gardening & Lawn Care / Soils, Fertilizers & Mulches / Fertilizers',
        'bbq grill':            'Garden & Outdoors / Grills & Outdoor Cooking / Grills / Charcoal Grills',
        'charcoal grill':       'Garden & Outdoors / Grills & Outdoor Cooking / Grills / Charcoal Grills',
        # Musical Instruments
        'acoustic guitar':      'Musical Instruments / Guitars & Basses / Acoustic Guitars / 6 String Guitars',
        'electric guitar':      'Musical Instruments / Guitars & Basses / Electric Guitars / Solid Body',
        'bass guitar':          'Musical Instruments / Guitars & Basses / Bass Guitars',
        'ukulele':              'Musical Instruments / Guitars & Basses / Ukuleles',
        'guitar strings':       'Musical Instruments / Instrument Accessories / Guitar & Bass Accessories / Strings / Acoustic Guitar Strings',
        'digital piano':        'Musical Instruments / Pianos, Keyboards & Organs / Digital Pianos',
        'drum set':             'Musical Instruments / Drums & Percussion / Acoustic Drums / Drum Sets',
        'violin':               'Musical Instruments / Orchestral Strings / Violins',
        'saxophone':            'Musical Instruments / Woodwinds / Saxophones',
        'guitar amplifier':     'Musical Instruments / Amplifiers & Effects / Guitar & Bass Amplifiers / Electric Guitar Amplifiers',
        # Industrial & Scientific
        'digital multimeter':   'Industrial & Scientific / Test, Measure & Inspect / Electrical Testing / Multimeters',
        'soldering iron':       'Industrial & Scientific / Professional & Industrial Equipment / Soldering & Desoldering / Soldering Irons',
        'angle grinder':        'Industrial & Scientific / Abrasive & Finishing Products / Abrasive Accessories / Angle Grinder Discs',
        'safety helmet':        'Industrial & Scientific / Safety Supplies / Head Protection / Hard Hats & Helmets',
        'safety vest':          'Industrial & Scientific / Safety Supplies / Protective Clothing / Safety Vests',
        'welding machine':      'Industrial & Scientific / Professional & Industrial Equipment / Welding & Soldering / Welding Equipment / MIG Welders',
        'air compressor':       'Industrial & Scientific / Hydraulics, Pneumatics & Plumbing / Pneumatics / Air Compressors',
        'water pump':           'Industrial & Scientific / Hydraulics, Pneumatics & Plumbing / Pumps & Pump Accessories / Water Pumps',
        'vernier caliper':      'Industrial & Scientific / Test, Measure & Inspect / Dimensional Measurement / Calipers',
        'fire extinguisher':    'Industrial & Scientific / Safety Supplies / Fire Safety / Fire Extinguishers',
        'toolbox set':          'Home & Office / Tools & Home Improvement / Power & Hand Tools / Hand Tools / Tool Storage / Tool Boxes',
        # Grocery
        'food storage':         'Grocery / Paper & Plastic / Disposable Food Storage / Food Storage Containers',
        'tomato stew':          'Grocery / Paper & Plastic / Disposable Food Storage / Food Storage Containers',
        'tomato paste':         'Grocery / Paper & Plastic / Disposable Food Storage / Food Storage Containers',
    }

    def load_learning_db(self):
        """Load learned product→category corrections from Firestore."""
        db = {}
        try:
            # Look for collection 'matcher_data', document 'learning_db'
            doc_ref = self.db_client.collection('matcher_data').document('learning_db')
            doc = doc_ref.get()
            
            if doc.exists:
                db_data = doc.to_dict()
                db = {k.lower(): v for k, v in db_data.items()}
        except Exception as e:
            print(f"🔥 FIREBASE LOAD ERROR: {e}")

        # Merge seed corrections — user corrections from the cloud always win
        for k, v in self._SEED_CORRECTIONS.items():
            if k not in db:
                db[k] = v
        return db

    def save_learning_db(self):
        """Persist the learning DB directly to Firestore."""
        try:
            doc_ref = self.db_client.collection('matcher_data').document('learning_db')
            doc_ref.set(self.learning_db)
            print("✅ Successfully synced learning_db to Firestore.")
        except Exception as e:
            print(f"🔥 FAILED TO SAVE TO FIREBASE: {e}")

    def export_learning_db(self) -> str:
        """
        Return the full learning DB as a pretty-printed JSON string.
        Use this to let users download the current corrections file from the
        Streamlit UI so it can be stored externally and re-imported later.
        """
        return json.dumps(self.learning_db, ensure_ascii=False, indent=2)

    def import_learning_db(self, json_str: str, merge: bool = True) -> int:
        """
        Load corrections from a JSON string (e.g. content of an uploaded file).
        """
        try:
            incoming = json.loads(json_str)
            if not isinstance(incoming, dict):
                return 0
            if merge:
                self.learning_db.update(incoming)
            else:
                self.learning_db = incoming
            self.save_learning_db()
            if SKLEARN_AVAILABLE and len(self.learning_db) >= 2:
                self._retrain_correction_classifier()
            return len(incoming)
        except Exception:
            return 0

    def apply_learned_correction(self, product_name, category):
        """
        Store a correction: product_name (lower) → category.
        Also retrains the sklearn SGDClassifier so the model immediately
        improves — future products similar to this one will route correctly.
        """
        self.learning_db[product_name.lower().strip()] = category
        self.save_learning_db()
        # Retrain the sklearn correction classifier in a background thread
        # so the UI stays responsive
        if SKLEARN_AVAILABLE:
            try:
                self._retrain_correction_classifier()
            except Exception:
                pass

    def lookup_learning_db(self, product_name):
        """
        Check the learning DB for an exact or near-exact match.
        Returns category string or None.
        """
        pn = product_name.lower().strip()
        if pn in self.learning_db:
            return self.learning_db[pn]
        for key, cat in self.learning_db.items():
            if len(key) >= 6:
                if key in pn or pn in key:
                    return cat
        return None

    def open_learning_panel(self):
        """Not available in engine mode (UI only in the desktop app)."""
        pass

    def _load_sklearn_model(self):
        """Load a previously saved sklearn model bundle from disk."""
        if not SKLEARN_AVAILABLE:
            return
        if os.path.exists(self._SK_MODEL_PATH):
            try:
                with open(self._SK_MODEL_PATH, 'rb') as f:
                    bundle = pickle.load(f)
                self._sk_clf        = bundle.get('clf')
                self._sk_clf_vec    = bundle.get('clf_vec')
                self._sk_le         = bundle.get('le')
                self._sk_clf_trained = bool(bundle.get('clf_trained', False))
            except Exception:
                pass  # Corrupt file — will rebuild fresh

    def _save_sklearn_model(self):
        """Persist the sklearn model bundle to disk."""
        if not SKLEARN_AVAILABLE or self._sk_clf is None:
            return
        try:
            bundle = {
                'clf':         self._sk_clf,
                'clf_vec':     self._sk_clf_vec,
                'le':          self._sk_le,
                'clf_trained': self._sk_clf_trained,
            }
            with open(self._SK_MODEL_PATH, 'wb') as f:
                pickle.dump(bundle, f)
        except Exception:
            pass

    # ── sklearn index building ─────────────────────────────────────────────

    @staticmethod
    def _sk_clean(text):
        """Normalise text for sklearn vectorisers."""
        return re.sub(r'[^a-z0-9\s]', ' ', str(text).lower()).strip()

    @staticmethod
    def _sk_sanitize(texts, placeholder='item'):
        """
        Ensure no text in the list is empty or whitespace-only.
        TfidfVectorizer raises "empty vocabulary" if ALL texts produce
        zero tokens — this prevents that by substituting a placeholder.
        """
        return [t if t.strip() else placeholder for t in texts]

    def _build_sklearn_index(self, categories_list):
        """
        Build sklearn TF-IDF indexes over all categories (called once per run).
        - Global matrix: all categories -> used as absolute fallback
        - Per-domain matrices: higher precision within each domain
        Fully defensive: sanitizes inputs so unusual text never causes
        "empty vocabulary" errors.
        """
        if not SKLEARN_AVAILABLE:
            return
        if self._sk_built and hasattr(self, '_sk_cats') and self._sk_cats == categories_list:
            return

        self._sk_cats = list(categories_list)

        def aug(c):
            """Full path + leaf repeated 3x for leaf-boosting."""
            try:
                leaf = str(c).split('/')[-1].strip()
                return self._sk_clean(c) + ' ' + (' '.join([self._sk_clean(leaf)] * 3))
            except Exception:
                return 'unknown category'

        # Sanitize: replace empty texts so vectorizer never gets an all-empty corpus
        aug_cats = self._sk_sanitize([aug(c) for c in categories_list], 'unknown')

        # Global vectoriser
        try:
            self._sk_global_vec = TfidfVectorizer(
                analyzer='word', ngram_range=(1, 2),
                sublinear_tf=True, min_df=1,
                max_features=80000,
                token_pattern=r'(?u)[a-z][a-z0-9]+',
            )
            self._sk_global_mat = self._sk_global_vec.fit_transform(aug_cats)
        except Exception:
            try:
                self._sk_global_vec = TfidfVectorizer(
                    analyzer='char_wb', ngram_range=(3, 5),
                    sublinear_tf=True, min_df=1, max_features=80000,
                )
                self._sk_global_mat = self._sk_global_vec.fit_transform(aug_cats)
            except Exception:
                self._sk_global_vec = None
                self._sk_global_mat = None

        # Per-domain vectorisers
        domain_to_cats = defaultdict(list)
        domain_to_aug  = defaultdict(list)
        for c, a in zip(categories_list, aug_cats):
            dom = str(c).split('/')[0].strip() or 'Other'
            domain_to_cats[dom].append(c)
            domain_to_aug[dom].append(a)

        self._sk_domain_vecs = {}
        for dom in domain_to_cats:
            try:
                sanitized = self._sk_sanitize(domain_to_aug[dom], 'unknown')
                dv = TfidfVectorizer(
                    analyzer='word', ngram_range=(1, 3),
                    sublinear_tf=True, min_df=1,
                    token_pattern=r'(?u)[a-z][a-z0-9]+',
                )
                dm = dv.fit_transform(sanitized)
                self._sk_domain_vecs[dom] = (dv, dm, domain_to_cats[dom])
            except Exception:
                pass  # Domain skipped; global fallback used for products in this domain

        self._sk_built = True

    def _sklearn_best_in_domain(self, product_name, domain_name):
        """
        Cosine similarity search within a single domain using sklearn.
        Returns (best_category, score).
        """
        if not self._sk_built or domain_name not in self._sk_domain_vecs:
            return None, 0.0
        try:
            dv, dm, dcats = self._sk_domain_vecs[domain_name]
            q  = self._sk_clean(product_name) or 'unknown'
            qv = dv.transform([q])
            sims = cosine_similarity(qv, dm)[0]
            idx  = int(sims.argmax())
            return dcats[idx], float(sims[idx])
        except Exception:
            return None, 0.0

    def _sklearn_global_best(self, product_name):
        """Global cosine similarity across all categories. Last-resort fallback."""
        if not self._sk_built or self._sk_global_mat is None:
            return None, 0.0
        try:
            q  = self._sk_clean(product_name) or 'unknown'
            qv = self._sk_global_vec.transform([q])
            sims = cosine_similarity(qv, self._sk_global_mat)[0]
            idx  = int(sims.argmax())
            return self._sk_cats[idx], float(sims[idx])
        except Exception:
            return None, 0.0

    # ── SGDClassifier: trains on corrections, improves over time ──────────

    def _retrain_correction_classifier(self):
        """
        (Re)train the SGDClassifier on all product→category corrections
        accumulated in learning_db.  Called after every new correction.
        Requires at least 2 distinct domain labels to train.
        """
        if not SKLEARN_AVAILABLE or len(self.learning_db) < 2:
            return

        products   = list(self.learning_db.keys())
        categories = list(self.learning_db.values())
        domains    = [c.split('/')[0].strip() for c in categories]

        if len(set(domains)) < 2:
            return  # Need at least 2 classes

        # Sanitize inputs — empty strings cause "empty vocabulary" in char_wb vectorizer
        X_texts = self._sk_sanitize(
            [self._sk_clean(p) for p in products], 'unknown product'
        )

        try:
            # Fit or re-fit the feature vectoriser on all corrections
            self._sk_clf_vec = TfidfVectorizer(
                analyzer='char_wb', ngram_range=(3, 5),
                sublinear_tf=True, min_df=1,
            )
            X = self._sk_clf_vec.fit_transform(X_texts)

            self._sk_le = LabelEncoder()
            y = self._sk_le.fit_transform(domains)

            self._sk_clf = SGDClassifier(
                loss='log_loss', max_iter=1000,
                random_state=42, class_weight='balanced',
            )
            self._sk_clf.fit(X, y)
            self._sk_clf_trained = True
            self._save_sklearn_model()
        except Exception:
            pass  # Silently skip if corpus still insufficient

    def _clf_predict_domain(self, product_name):
        """
        Use the trained SGDClassifier to predict the most likely domain.
        Returns (domain_name, confidence) or (None, 0.0) if not trained.
        """
        if (not SKLEARN_AVAILABLE or not self._sk_clf_trained
                or self._sk_clf is None or self._sk_clf_vec is None):
            return None, 0.0
        try:
            qv    = self._sk_clf_vec.transform([self._sk_clean(product_name)])
            probs = self._sk_clf.predict_proba(qv)[0]
            idx   = int(np.argmax(probs))
            dom   = self._sk_le.classes_[idx]
            return dom, float(probs[idx])
        except Exception:
            return None, 0.0

    # ── Domain routing (keyword-based — primary, fast) ─────────────────────

    def _route_domain(self, product_name):
        """
        Route a product to its top-level domain using keyword phrase matching.
        Returns (domain_name, score) or (None, 0.0).
        """
        _single_word_veto = {
            'boost', 'heater', 'strip', 'strips', 'oil', 'bottle',
            'spray', 'guard', 'tablet', 'tablets', 'powder',
        }
        pn = product_name.lower()
        scores = defaultdict(float)
        for domain, kws in self._DOMAIN_KEYWORDS.items():
            for kw in kws:
                if kw in pn:
                    if kw.strip() in _single_word_veto:
                        continue
                    scores[domain] += len(kw.split()) * 2.0
        if not scores:
            return None, 0.0
        best = max(scores, key=scores.get)
        return best, scores[best]

    def _map_product_type(self, product_name):
        """
        Exact product-type lookup in _PRODUCT_CATEGORY_MAP.
        Returns the mapped category path, or None.
        Longest-match wins to avoid short-key false positives.
        """
        pn = product_name.lower()
        best_cat, best_len = None, 0
        for key, cat in self._PRODUCT_CATEGORY_MAP.items():
            if key in pn and len(key) > best_len:
                best_len = len(key)
                best_cat = cat

        if best_cat and 'phones & tablets' in best_cat.lower():
            _health_signals = {
                'mg', 'mcg', 'weight', 'supplement', 'vitamin', 'capsule',
                'herbal', 'medicine', 'health', 'immune', 'digestive',
                'blocker', 'loss', 'burn', 'detox', 'natural', 'organic',
                'dose', 'dosage', 'nutrient',
            }
            if any(sig in pn for sig in _health_signals):
                best_cat = None

        return best_cat

    # ── Legacy manual TF-IDF (fallback when sklearn unavailable) ──────────

    def _build_domain_indexes(self, categories_list):
        """Build per-domain manual TF-IDF indexes (legacy path)."""
        if (self._domain_indexes and self._domain_index_cats == categories_list):
            return
        self._domain_index_cats = list(categories_list)
        domain_cats_map = defaultdict(list)
        for c in categories_list:
            domain = c.split('/')[0].strip()
            domain_cats_map[domain].append(c)
        self._domain_indexes = {}
        for domain, dcats in domain_cats_map.items():
            n = len(dcats)
            doc_freqs, df_counts = [], Counter()
            for cat in dcats:
                tokens = self._tokenize(cat)
                tf = Counter(tokens)
                doc_freqs.append(tf)
                df_counts.update(set(tokens))
            idf = {t: math.log((n+1)/(f+1))+1.0 for t, f in df_counts.items()}
            self._domain_indexes[domain] = (dcats, doc_freqs, idf)

    def _tokenize(self, text):
        """Tokenize text into unigrams + bigrams."""
        text   = text.lower()
        tokens = re.findall(r'[a-z0-9]+', text)
        bigrams = [tokens[i]+'_'+tokens[i+1] for i in range(len(tokens)-1)]
        return tokens + bigrams

    def _tfidf_in_domain(self, product_name, domain_name, boost_path=None):
        """Manual TF-IDF within a domain (legacy fallback)."""
        if not self._domain_indexes or domain_name not in self._domain_indexes:
            return None, 0.0
        dcats, doc_freqs, idf = self._domain_indexes[domain_name]
        qtokens = self._tokenize(product_name)
        if boost_path:
            qtokens += self._tokenize(boost_path) * 3
        best_score, best_idx = 0.0, 0
        for i, dtf in enumerate(doc_freqs):
            score = sum(dtf[t] * idf.get(t, 0) for t in qtokens if t in dtf)
            if score > best_score:
                best_score = score
                best_idx   = i
        return (dcats[best_idx] if best_score > 0 else dcats[0]), best_score

    # ── Public index builder (called by run_matching) ──────────────────────

    def build_tfidf_index(self, categories_list):
        """Build the full similarity index (sklearn or legacy math)."""
        if SKLEARN_AVAILABLE:
            self._build_sklearn_index(categories_list)
        else:
            self._build_domain_indexes(categories_list)
        self._tfidf_built = True

    # ── Similarity match (used as fallback in pipeline) ───────────────────

    def similarity_match(self, product_name, categories_list):
        """
        Find the best category using sklearn (or legacy TF-IDF).
        """
        # ── Candidate domains ──────────────────────────────────────────────
        clf_dom, clf_conf = self._clf_predict_domain(product_name)
        kw_dom,  kw_score = self._route_domain(product_name)
        known_path        = self._map_product_type(product_name)

        # Build ordered list of domains to search (no duplicates)
        candidates = []
        if kw_dom and kw_score >= 4.0:
            candidates.append(kw_dom)
        if clf_dom and clf_dom not in candidates:
            # Insert classifier result: after keyword domain if confident, else first
            if clf_conf > 0.5 and not candidates:
                candidates.insert(0, clf_dom)
            else:
                candidates.append(clf_dom)
        if kw_dom and kw_dom not in candidates:
            candidates.append(kw_dom)

        # ── Search domains ─────────────────────────────────────────────────
        best_cat, best_score = None, -1.0

        if SKLEARN_AVAILABLE and self._sk_built:
            for dom in candidates[:3]:
                cat, sc = self._sklearn_best_in_domain(product_name, dom)
                if cat and sc > best_score:
                    best_score = sc
                    best_cat   = cat
            # Global fallback
            if best_cat is None or best_score < 0.05:
                best_cat, best_score = self._sklearn_global_best(product_name)
        else:
            # Legacy path
            if not self._domain_indexes:
                self._build_domain_indexes(categories_list)
            for dom in candidates[:3]:
                cat, sc = self._tfidf_in_domain(product_name, dom, boost_path=known_path)
                if cat and sc > best_score:
                    best_score = sc
                    best_cat   = cat
            if best_cat is None:
                for dom in self._domain_indexes:
                    cat, sc = self._tfidf_in_domain(product_name, dom, boost_path=known_path)
                    if sc > best_score:
                        best_score = sc
                        best_cat   = cat

        return (best_cat or categories_list[0], best_score)

    # ── Master matching pipeline ──────────────────────────────────────────

    def get_category_with_fallback(self, product_name, keyword_mapping,
                                   categories_list, leaf_categories=None,
                                   last_category_parts=None):
        """
        Five-priority matching pipeline called for every product.
        """
        if not product_name or isinstance(product_name, float):
            return categories_list[0] if categories_list else 'Uncategorized'

        # ── 0. Gender-lock pre-filter ─────────────────────────────────────
        _women_signals = [
            'ladies','for ladies','for women','women ','womens',
            'female ','girl wear','ladies wear','ladies top','ladies pant',
            'ladies bra','ladies gown','ladies dress','ladies night','ladies sleep',
            'ladies panties','ladies underwear','ladies tight','ladies camisole',
            'ladies singlet','ladies kaftan','ladies sexy','ladies beautiful',
            'ladies lovely','ladies classy','ladies elegant','ladies push up',
            'ladies stretchable','ladies quality','ladies casual',
            'nightwear for ladies','sleepwear for ladies','nightwear for women',
            'night wear for ladies','nightgown for ladies','bumper tight',
            'biker short','sexy nightwear','beautiful nightwear','classy nightwear',
            'lovely nightwear','elegant nightwear','sexy sleepwear','g-string ladies',
            'gstring ladies','boyshorts','female pant','female tight',
            'sexy ladies','sexy yoga pant','sexy yoga pants',
        ]
        _wrong_domains_for_women = {
            'baby products', 'automobile', 'books, movies and music',
            'home & office', 'industrial & scientific', 'computing',
            'grocery', 'garden & outdoors', 'pet supplies', 'sporting goods',
        }
        _pn_lower = product_name.lower()
        _has_women_signal = any(sig in _pn_lower for sig in _women_signals)

        # ── 1. Learning DB ────────────────────────────────────────────────
        learned = self.lookup_learning_db(product_name)
        if learned:
            if learned in categories_list:
                return learned
            ll = learned.lower()
            for cat in categories_list:
                if cat.lower() == ll:
                    return cat

        # ── 2. Exact product-type map ─────────────────────────────────────
        mapped = self._map_product_type(product_name)
        if mapped:
            if mapped in categories_list:
                return mapped
            for cat in categories_list:
                if cat.lower() == mapped.lower():
                    return cat
            parts = mapped.split('/')
            while len(parts) > 1:
                prefix = '/'.join(parts).lower().strip()
                for cat in categories_list:
                    if cat.lower().strip().startswith(prefix):
                        return cat
                parts = parts[:-1]

        # ── 3. Rule-based v2 engine ───────────────────────────────────────
        rule_result = self.get_category_for_product_v2(
            product_name, keyword_mapping, categories_list,
            leaf_categories, last_category_parts
        )

        # ── 4 & 5. Similarity verification / fallback ─────────────────────
        stopwords = {
            'and','the','for','with','new','set','pack','pcs','best','top',
            'pro','kit','use','per','all','our','high','quality','inch','size',
            'style','type','mini','large','small','super','ultra','max','plus',
            'big','pair','2in1','series','model','brand','version','color',
        }
        pwords  = set(re.findall(r'[a-z]{3,}', product_name.lower())) - stopwords
        rwords  = set(re.findall(r'[a-z]{3,}', rule_result.lower()))  - stopwords
        overlap = pwords & rwords

        if len(overlap) >= 1:
            candidate = rule_result  # Rule engine result looks plausible
        else:
            sim_result, sim_score = self.similarity_match(product_name, categories_list)
            if sim_score >= 0.05:
                candidate = sim_result
            else:
                candidate = rule_result

        # ── Gender-lock post-filter ───────────────────────────────────────
        if _has_women_signal:
            candidate_dom = candidate.split('/')[0].strip().lower()
            if candidate_dom in _wrong_domains_for_women:
                candidate = self._womens_fashion_fallback(product_name, categories_list)
            elif "men's fashion" in candidate.lower() or 'mens fashion' in candidate.lower():
                candidate = self._womens_fashion_fallback(product_name, categories_list)

        return candidate

    def _womens_fashion_fallback(self, product_name, categories_list):
        """
        Return the most appropriate Women's Fashion category for a product
        that was wrongly routed away from Fashion.
        """
        pn = product_name.lower()

        _WOMENS_RULES = [
            (['bra', 'push up bra', 'breathable bra', 'cross bra', 'flower bra'],
             "Fashion / Women's Fashion / Underwear & Sleepwear / Bra"),
            (['bumper tight', 'biker short', 'legging', ' tight ', 'sport tight', 'yoga pant', 'yoga pants'],
             "Fashion / Women's Fashion / Clothing / Leggings"),
            (['sexy nightgown', 'ladies nightgown', 'sexy gown', 'night gown', 'nightgown'],
             "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Nightgowns & Sleepshirts"),
            (['nightwear', 'sleepwear', 'night wear', 'sleep wear', 'pyjama', 'pyjamas'],
             "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Sets"),
            (['panties', 'underwear', 'pant ', 'g-string', 'gstring', 'boyshort', 'briefs'],
             "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Lingerie / Panties"),
            (['lingerie', 'camisole', 'singlet top', 'bratop'],
             "Fashion / Women's Fashion / Clothing / Lingerie, Sleep & Lounge / Lingerie / Camisoles & Tanks"),
            (['gown', 'dress', 'kaftan', 'abaya', 'kimono', 'caftan', 'maxi', 'abaya'],
             "Fashion / Women's Fashion / Clothing / Dresses"),
            (['jean', 'jeans', 'denim'],
             "Fashion / Women's Fashion / Clothing / Jeans"),
            (['shorts'],
             "Fashion / Women's Fashion / Clothing / Shorts"),
            (['top ', ' top', 'blouse', 'shirt', 'singlet'],
             "Fashion / Women's Fashion / Clothing / Tops & Tees"),
        ]

        for keywords, target_cat in _WOMENS_RULES:
            if any(kw in pn for kw in keywords):
                if target_cat in categories_list:
                    return target_cat
                prefix = target_cat.split('/')[:-1]
                while len(prefix) > 2:
                    p = '/'.join(prefix).lower().strip()
                    for cat in categories_list:
                        if cat.lower().strip().startswith(p) and "women" in cat.lower():
                            return cat
                    prefix = prefix[:-1]

        for cat in categories_list:
            if "Fashion / Women's Fashion / Clothing" in cat:
                return cat
        for cat in categories_list:
            if "Women's Fashion" in cat or "Womens Fashion" in cat:
                return cat
        return "Fashion / Women's Fashion / Clothing / Tops & Tees"

    def clean_text(self, text):
        """Clean and normalize text for matching"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        return re.sub(r'\W+', ' ', text.lower().strip())
    
    def extract_keywords(self, text, max_keywords=15):
        """Extract meaningful keywords from text"""
        if not text:
            return []
        cleaned = self.clean_text(text)
        words = cleaned.split()
        keywords = [w for w in words if len(w) > 2 and not w.isdigit()]
        return keywords[:max_keywords]
    
    def identify_product_type(self, product_name):
        """
        Identify the exact product type from the product name.
        """
        if pd.isna(product_name) or not isinstance(product_name, str):
            return None, []
        
        product_lower = product_name.lower()
        
        ignore_patterns = [
            r'\b(lg|samsung|sony|panasonic|philips|whirlpool|haier|sharp|bosch|siemens|electrolux|daewoo|hitachi|toshiba|mitsubishi|canon|nikon|apple|samsung|google|microsoft|amazon)\b',
            r'\b(1\.?5hp|2hp|1hp|0\.?5hp|\d+hp)\b',
            r'\b\d+\s*(pieces|pcs|piece|set|pack|pairs|boxes|units|kg|gram|g|lb|oz|ml|l|cm|mm|inch|inches|feet|foot|meter)\b',
            r'\b(premium|professional|commercial|industrial|household|portable|compact|mini|compact|wireless|bluetooth|smart|automatic|manual|electric|non-electric)\b',
            r'\b(new|latest|newest|best|seller|top|popular|featured|deal|offer|free|fast|same day|express|overnight)\b',
            r'\b(white|black|silver|gray|grey|gold|rose gold|pink|blue|red|green|yellow|orange|purple|brown|beige|multicolor)\b',
            r'\b(gift|套装|combo|bundle|deal|pack|box|packaging|wrapping|bag|wrap)\b',
            r'\b(genuine|original|authentic|imported|export|local|custom|made\s*in)\b',
            r'\b(\d{4}|\d{2}-\d{2}|v\d+\.?\d*|model\s*\w+)\b',
        ]
        
        cleaned_name = product_lower
        for pattern in ignore_patterns:
            cleaned_name = re.sub(pattern, ' ', cleaned_name)
        
        cleaned_name = re.sub(r'\s+', ' ', cleaned_name).strip()
        
        women_fashion_patterns = [
            (r'\bnightwear\b|\bnight\s*wear\b|\bsleepwear\b|\bsleep\s*wear\b|\bpyjama\b|\bpajama\b', 'Nightwear'),
            (r'\bnightgown\b|\bnight\s*gown\b|\bnightie\b', 'Nightgown'),
            (r'\bpanties\b|\bpantie\b|\bpanty\b|\bunderwear\b|\bundergarment\b|\bg.?string\b|\bthong\b|\bbriefs?\b(?!.*sport|.*cycle)', 'Women Panties'),
            (r'\bbra\b|\bbras\b|\bbralette\b|\bbratop\b|\bpush.?up\s*bra\b', 'Women Bra'),
            (r'\blingerie\b|\bsexy\s*lace\b|\bsexy\s*underwear\b', 'Women Lingerie'),
            (r'\bshapewear\b|\bgirdle\b|\bbody\s*shaper\b|\btummy\s*tight\b|\bslimming\b', 'Women Shapewear'),
            (r'\bcamisole\b|\bcami\b|\bsinglet\b(?!.*men)', 'Women Camisole'),
            (r'\bleggings?\b|\btight\b|\btights\b(?!.*tool|.*jeans)', 'Women Leggings'),
            (r'\bjeans?\b(?!.*men)', 'Women Jeans'),
            (r'\bkaftan\b|\bcaftan\b|\babaya\b|\bbubu\b|\bkimono\b(?!.*jacket|.*robe\s*men)', 'Women Dress'),
            (r'\bgown\b|\bdress\b(?!.*code|.*shirt)', 'Women Dress'),
            (r'\btop\s+and\s+(short|trouser|pant)\b|\bshirt\s+and\s+(trouser|pant|short)\b', 'Women Tops Set'),
            (r'\bshort\b|\bshorts\b(?!.*circuit|.*cut)', 'Women Shorts'),
        ]
        
        for pattern, product_type in women_fashion_patterns:
            if re.search(pattern, product_lower, re.IGNORECASE):
                return product_type, self._get_context_keywords(product_name, product_type)

        men_fashion_patterns = [
            (r'\bmen\s*jeans?\b|\bguy\s*jeans?\b', 'Men Jeans'),
            (r'\bmen\s*trouser\b|\bmen\s*pants?\b|\bguy\s*trouser\b', 'Men Trousers'),
            (r'\bmen\s*shorts?\b|\bguy\s*shorts?\b', 'Men Shorts'),
            (r'\bmen\s*shirt\b|\bguy\s*shirt\b|\bmen\s*polo\b', 'Men Shirt'),
            (r'\bmen\s*suit\b|\bguy\s*suit\b|\bmen\s*blazer\b', 'Men Suit'),
            (r'\bmen\s*jacket\b|\bmen\s*coat\b|\bguy\s*jacket\b', 'Men Jacket'),
            (r'\bmen\s*hoodie\b|\bmen\s*sweatshirt\b', 'Men Hoodie'),
            (r'\bmen\s*underwear\b|\bmen\s*boxer\b|\bmen\s*brief\b', 'Men Underwear'),
            (r'\bmen\s*sock\b|\bmen\s*socks\b', 'Men Socks'),
            (r'\bmen\s*cap\b|\bmen\s*hat\b|\bguy\s*cap\b', 'Men Cap'),
            (r'\bmen\s*belt\b', 'Men Belt'),
            (r'\bmen\s*tie\b|\bnecktie\b', 'Men Tie'),
            (r'\bmen\s*wallet\b', 'Men Wallet'),
            (r'\bmen\s*wrist\s*watch\b|\bmen\s*watch\b|\bgent\s*watch\b', 'Men Watch'),
            (r'\bmen\s*t.?shirt\b|\bguy\s*t.?shirt\b', 'Men T-Shirt'),
            (r'\bmen\s*singlet\b|\bguy\s*singlet\b', 'Men Singlet'),
            (r'\bmen\s*shoe\b|\bguy\s*shoe\b|\bmen\s*sneaker\b', 'Men Shoes'),
            (r'\bmen\s*sandal\b|\bguy\s*sandal\b', 'Men Sandals'),
            (r'\bmen\s*boot\b|\bguy\s*boot\b', 'Men Boots'),
            (r'\bmen\s*fabric\b|\bguy\s*fabric\b', 'Men Fabric'),
            (r'\bagbada\b|\bdaishiki\b|\bsenator\b(?!.*pen|.*bank)', 'African Men Wear'),
            (r'\bmen\s*pyjama\b|\bmen\s*pajama\b|\bmen\s*sleepwear\b', 'Men Sleepwear'),
        ]
        for pattern, product_type in men_fashion_patterns:
            if re.search(pattern, product_lower, re.IGNORECASE):
                return product_type, self._get_context_keywords(product_name, product_type)

        kids_fashion_patterns = [
            (r'\bgirl\s*dress\b|\bgirls\s*dress\b|\bgirl\s*gown\b', 'Girls Dress'),
            (r'\bgirl\s*skirt\b|\bgirls\s*skirt\b', 'Girls Skirt'),
            (r'\bgirl\s*top\b|\bgirls\s*top\b|\bgirl\s*blouse\b', 'Girls Top'),
            (r'\bboy\s*shirt\b|\bboys\s*shirt\b|\bboy\s*top\b', 'Boys Shirt'),
            (r'\bboy\s*trouser\b|\bboys\s*trouser\b|\bboy\s*pant\b', 'Boys Trousers'),
            (r'\bboy\s*short\b|\bboys\s*short\b', 'Boys Shorts'),
            (r'\bkid\s*shoe\b|\bkids\s*shoe\b|\bchildren\s*shoe\b|\binfant\s*shoe\b|\bbaby\s*shoe\b', 'Kids Shoes'),
            (r'\bschool\s*uniform\b|\bschool\s*wear\b', 'School Uniform'),
            (r'\bkid\s*cloth\b|\bkids\s*cloth\b|\bchildren\s*cloth\b|\bbaby\s*cloth\b|\btoddler\s*cloth\b', 'Kids Clothing'),
            (r'\bkid\s*sock\b|\bkids\s*sock\b|\bbaby\s*sock\b', 'Kids Socks'),
            (r'\bkid\s*hat\b|\bkids\s*hat\b|\bbaby\s*hat\b|\bkid\s*cap\b', 'Kids Cap'),
            (r'\bkid\s*pyjama\b|\bkids\s*pyjama\b|\bbaby\s*pyjama\b|\bchildren\s*pyjama\b', 'Kids Sleepwear'),
        ]
        for pattern, product_type in kids_fashion_patterns:
            if re.search(pattern, product_lower, re.IGNORECASE):
                return product_type, self._get_context_keywords(product_name, product_type)

        traditional_patterns = [
            (r'\bankara\b(?!.*fabric|.*cloth)', 'Ankara Wear'),
            (r'\btraditional\s*wear\b|\btraditional\s*cloth\b|\bafrican\s*wear\b|\bafrican\s*cloth\b', 'Traditional Wear'),
            (r'\baso.?oke\b', 'Aso-Oke Fabric'),
            (r'\bkente\b', 'Traditional Wear'),
            (r'\bkidagba\b|\bbubou\b|\bgele\b', 'Traditional Wear'),
        ]
        for pattern, product_type in traditional_patterns:
            if re.search(pattern, product_lower, re.IGNORECASE):
                return product_type, self._get_context_keywords(product_name, product_type)

        electronics_patterns = [
            (r'\bsmart\s*tv\b|\bled\s*tv\b|\boled\s*tv\b|\b4k\s*tv\b|\buhd\s*tv\b|\blcd\s*tv\b|\btelevision\b|\bflat\s*screen\b', 'Television'),
            (r'\bsound\s*bar\b|\bsoundbar\b', 'Sound Bar'),
            (r'\bhome\s*theater\b|\bhome\s*theatre\b', 'Home Theater'),
            (r'\bbluetooth\s*speaker\b|\bportable\s*speaker\b|\bwireless\s*speaker\b|\bspeaker\b(?!.*phone)', 'Speaker'),
            (r'\bheadphone\b|\bover.?ear\s*headphone\b', 'Headphones'),
            (r'\bwireless\s*earbud\b|\bbluetooth\s*earbud\b|\btws\s*earbud\b|\bearphone\b|\bairpod\b', 'Earphones'),
            (r'\bwireless\s*headset\b|\bBluetooth\s*headset\b', 'Headset'),
            (r'\bdslr\b|\bmirrorless\s*camera\b|\bdigital\s*camera\b|\binstax\b|\bpolaroid\s*camera\b', 'Camera'),
            (r'\baction\s*camera\b|\bgopro\b|\bbody\s*camera\b', 'Action Camera'),
            (r'\bsecurity\s*camera\b|\bcctv\b|\bip\s*camera\b|\bsurveillance\s*camera\b', 'Security Camera'),
            (r'\bweb\s*cam\b|\bwebcam\b', 'Webcam'),
            (r'\bprojector\b|\bdigital\s*projector\b|\bmini\s*projector\b', 'Projector'),
            (r'\bgps\s*tracker\b|\bgps\s*device\b|\bnavigation\s*device\b', 'GPS Device'),
            (r'\bwalkie\s*talkie\b|\btwo.?way\s*radio\b', 'Two-Way Radio'),
            (r'\bportable\s*radio\b|\bam\s*fm\s*radio\b|\bdigital\s*radio\b', 'Radio'),
            (r'\bdvd\s*player\b|\bblu.?ray\s*player\b', 'DVD Player'),
            (r'\bstreambox\b|\bstreaming\s*device\b|\bfiretv\b|\bandroid\s*tv\s*box\b', 'Streaming Device'),
            (r'\bvr\s*headset\b|\bvirtual\s*reality\b', 'VR Headset'),
            (r'\bdrone\b(?!.*rc)', 'Drone'),
        ]
        for pattern, product_type in electronics_patterns:
            if re.search(pattern, product_lower, re.IGNORECASE):
                return product_type, self._get_context_keywords(product_name, product_type)

        home_patterns = [
            (r'\bsofa\b|\bcouch\b|\bsectional\s*sofa\b|\bsofa\s*set\b', 'Sofa'),
            (r'\bbed\s*frame\b|\bbunk\s*bed\b|\bdouble\s*bed\b|\bking\s*bed\b|\bqueen\s*bed\b', 'Bed Frame'),
            (r'\bwardrobe\b|\bcloset\b|\bdressing\b(?!.*table)', 'Wardrobe'),
            (r'\bdressing\s*table\b|\bvanity\s*table\b', 'Dressing Table'),
            (r'\bcoffee\s*table\b|\bside\s*table\b|\baccent\s*table\b', 'Side Table'),
            (r'\bdining\s*table\b|\bdining\s*set\b(?!.*cookware|.*kitchen)', 'Dining Table'),
            (r'\btv\s*stand\b|\btv\s*console\b|\bentertainment\s*stand\b', 'TV Stand'),
            (r'\bmirror\b(?!.*car|.*side|.*rear)', 'Mirror'),
            (r'\bwall\s*art\b|\bcanvas\s*art\b|\bwall\s*painting\b|\bwall\s*decor\b', 'Wall Art'),
            (r'\bwall\s*clock\b|\bdesk\s*clock\b|\btable\s*clock\b', 'Clock'),
            (r'\bvase\b|\bflower\s*vase\b', 'Vase'),
            (r'\bphoto\s*frame\b|\bpicture\s*frame\b', 'Photo Frame'),
            (r'\bcurtain\b|\bwindow\s*blind\b|\bblackout\s*curtain\b', 'Curtain'),
            (r'\btable\s*runner\b|\btablecloth\b|\btable\s*cloth\b', 'Tablecloth'),
            (r'\bscent\s*diffuser\b|\broad\s*diffuser\b|\baromatherapy\b(?!.*oil)', 'Diffuser'),
            (r'\bdesk\s*lamp\b|\btable\s*lamp\b|\bfloor\s*lamp\b|\blamp\s*shade\b', 'Lamp'),
            (r'\bceiling\s*light\b|\bpendant\s*light\b|\bchandelier\b', 'Ceiling Light'),
            (r'\bled\s*strip\b|\bled\s*strip\s*light\b|\brgb\s*strip\b', 'LED Strip Light'),
            (r'\bsolar\s*panel\b|\bsolar\s*system\b|\bsolar\s*inverter\b', 'Solar Panel'),
            (r'\binverter\b(?!.*solar|.*car)', 'Inverter'),
            (r'\bups\b(?!.*tracking)', 'UPS Battery Backup'),
            (r'\bvacuum\s*cleaner\b|\bhoover\b|\bcordless\s*vacuum\b', 'Vacuum Cleaner'),
            (r'\biron\b(?!.*protein|.*supplement|.*vitamin)|\bsteam\s*iron\b|\bclothes\s*iron\b', 'Clothes Iron'),
            (r'\bsewing\s*machine\b', 'Sewing Machine'),
            (r'\bknitting\s*needle\b|\bcrochet\s*hook\b', 'Knitting Needles'),
            (r'\bstaple\s*gun\b|\bstapler\b', 'Stapler'),
            (r'\bshredder\b|\bpaper\s*shredder\b', 'Paper Shredder'),
            (r'\blaminator\b|\blaminat\b', 'Laminator'),
            (r'\bthermal\s*printer\b|\blabel\s*printer\b', 'Label Printer'),
        ]
        for pattern, product_type in home_patterns:
            if re.search(pattern, product_lower, re.IGNORECASE):
                return product_type, self._get_context_keywords(product_name, product_type)

        health_patterns = [
            (r'\bhair\s*extension\b|\bhair\s*weave\b|\bbundle\s*hair\b|\bclosure\s*hair\b|\bfrontal\s*hair\b', 'Hair Extension'),
            (r'\bwig\b|\blace\s*front\s*wig\b|\bhuman\s*hair\s*wig\b|\bsynth\w*\s*wig\b', 'Wig'),
            (r'\bhair\s*attachment\b|\bbraiding\s*hair\b|\btwist\s*hair\b|\bdread\s*lock\b', 'Hair Attachment'),
            (r'\bhair\s*dryer\b|\bblowdryer\b|\bhair\s*blow\b', 'Hair Dryer'),
            (r'\bhair\s*straightener\b|\bflat\s*iron\b|\bhair\s*flat\b', 'Hair Straightener'),
            (r'\bhair\s*curler\b|\bcurling\s*iron\b|\bcurling\s*wand\b', 'Hair Curler'),
            (r'\bnail\s*polish\b|\bnailvarnish\b|\bnail\s*gel\b', 'Nail Polish'),
            (r'\bnail\s*drill\b|\bnail\s*lamp\b|\bnail\s*kit\b', 'Nail Kit'),
            (r'\blip\s*stick\b|\blipstick\b|\blip\s*gloss\b|\blip\s*liner\b|\blip\s*balm\b', 'Lipstick'),
            (r'\bfoundation\b(?!.*underwear)|\bbb\s*cream\b|\bcc\s*cream\b', 'Foundation'),
            (r'\bconcealer\b', 'Concealer'),
            (r'\bhighlighter\b(?!.*pen|.*marker)', 'Highlighter Makeup'),
            (r'\bblush\b(?!.*wine)|\bblusher\b', 'Blush'),
            (r'\beyeliner\b|\bkohl\b|\beye\s*liner\b', 'Eyeliner'),
            (r'\bmascaral?\b|\beye\s*mascara\b', 'Mascara'),
            (r'\beyeshadow\b|\beye\s*shadow\b|\beyeshadow\s*palette\b', 'Eyeshadow'),
            (r'\bmakeup\s*brush\b|\bblend\w*\s*sponge\b|\bmakeup\s*kit\b', 'Makeup Brush'),
            (r'\bdeodorant\b|\broller\s*deodorant\b|\broll.?on\b', 'Deodorant'),
            (r'\bbody\s*spray\b|\bcologne\b', 'Body Spray'),
            (r'\belectric\s*shaver\b|\bshaving\s*machine\b|\bbarbering\s*clipper\b|\bhair\s*clipper\b|\btrimmer\b', 'Hair Clipper'),
            (r'\belectric\s*razor\b|\bwet\s*shaver\b', 'Electric Razor'),
            (r'\bblood\s*pressure\b|\bbp\s*monitor\b|\bsphygmo\b', 'Blood Pressure Monitor'),
            (r'\bblood\s*glucose\b|\bglucometer\b|\bdiabetes\s*monitor\b', 'Glucometer'),
            (r'\bpulse\s*oximeter\b|\bspo2\b', 'Pulse Oximeter'),
            (r'\bthermometer\b|\bdigital\s*thermometer\b|\binfra.?red\s*thermometer\b', 'Thermometer'),
            (r'\bback\s*brace\b|\bback\s*support\b|\blumbar\s*support\b', 'Back Support'),
            (r'\bcompression\s*sock\b|\bmedical\s*sock\b', 'Compression Socks'),
            (r'\bwheelchair\b', 'Wheelchair'),
            (r'\bwalker\b(?!.*phone|.*android)', 'Walking Aid'),
            (r'\bcrutch\b', 'Crutches'),
            (r'\bsanitary\s*pad\b|\bmenstrual\s*pad\b|\bpads?\b(?!.*knee|.*shoulder|.*elbow|.*anti)', 'Sanitary Pad'),
            (r'\btampon\b', 'Tampon'),
            (r'\bfeminine\s*wash\b|\bintimate\s*wash\b', 'Feminine Wash'),
            (r'\bcondom\b(?!.*pant)', 'Condom'),
            (r'\blubricant\b(?!.*car|.*machine)', 'Lubricant'),
            (r'\bmassage\s*gun\b|\bpercussion\s*massager\b', 'Massage Gun'),
            (r'\bmassage\s*chair\b|\bfull\s*body\s*massager\b', 'Massage Chair'),
            (r'\belectric\s*massager\b|\bvibration\s*massager\b|\bhandheld\s*massager\b', 'Electric Massager'),
            (r'\bweight\s*loss\b|\bfat\s*burn\b|\bslimming\s*tea\b|\bdetox\s*tea\b', 'Weight Loss Supplement'),
            (r'\bprotein\s*shake\b|\bwhey\s*protein\b|\bprotein\s*powder\b', 'Protein Powder'),
            (r'\bcreatine\b', 'Sports Supplement'),
            (r'\bomega.?3\b|\bfish\s*oil\b', 'Fish Oil'),
            (r'\bcollagen\b', 'Collagen Supplement'),
            (r'\bglutathione\b', 'Skin Supplement'),
        ]
        for pattern, product_type in health_patterns:
            if re.search(pattern, product_lower, re.IGNORECASE):
                return product_type, self._get_context_keywords(product_name, product_type)

        auto_patterns = [
            (r'\bcar\s*wash\b|\bcar\s*shampoo\b|\bauto\s*wash\b', 'Car Wash'),
            (r'\bcar\s*wax\b|\bcar\s*polish\b|\bauto\s*polish\b', 'Car Polish'),
            (r'\bcar\s*seat\s*cover\b|\bauto\s*seat\s*cover\b', 'Car Seat Cover'),
            (r'\bcar\s*floor\s*mat\b|\bauto\s*floor\s*mat\b|\bcar\s*mat\b', 'Car Floor Mat'),
            (r'\bcar\s*cover\b|\bvehicle\s*cover\b|\bauto\s*cover\b', 'Car Cover'),
            (r'\bcar\s*vacuum\b|\bportable\s*car\s*vacuum\b', 'Car Vacuum'),
            (r'\bcar\s*air\s*freshener\b|\bauto\s*freshener\b', 'Car Air Freshener'),
            (r'\bcar\s*charger\b|\bauto\s*charger\b|\bvehicle\s*charger\b', 'Car Charger'),
            (r'\bdash\s*cam\b|\bdashboard\s*camera\b|\bcar\s*camera\b', 'Dash Cam'),
            (r'\btyre\b|\btire\b(?!.*pressure)', 'Tyre'),
            (r'\bwheel\s*cap\b|\bhub\s*cap\b|\bwheel\s*cover\b', 'Wheel Cap'),
            (r'\bbrake\s*pad\b|\bbrake\s*disc\b|\bbrake\s*drum\b', 'Brake Parts'),
            (r'\bengine\s*oil\b|\bmotor\s*oil\b|\bsynth\w*\s*oil\b', 'Motor Oil'),
            (r'\btransmission\s*fluid\b|\bgear\s*oil\b', 'Transmission Fluid'),
            (r'\bcar\s*battery\b|\bauto\s*battery\b', 'Car Battery'),
            (r'\bcar\s*radio\b|\bhead\s*unit\b|\bcar\s*stereo\b|\bcar\s*dvd\b', 'Car Stereo'),
            (r'\bcar\s*light\b|\bcar\s*led\b|\bheadlamp\b|\bheadlight\b|\btaillight\b|\btail\s*lamp\b', 'Car Light'),
            (r'\bparking\s*sensor\b|\breverse\s*sensor\b|\bpdc\b(?!.*sensor)', 'Parking Sensor'),
            (r'\bcar\s*jack\b|\bhydraulic\s*jack\b|\bscrew\s*jack\b', 'Car Jack'),
            (r'\btire\s*inflator\b|\bair\s*compressor\b(?!.*industrial)', 'Tyre Inflator'),
            (r'\bcar\s*lock\b|\bsteering\s*lock\b', 'Car Security'),
            (r'\bcar\s*alarm\b|\bauto\s*alarm\b', 'Car Alarm'),
            (r'\bmotorcycle\b|\bbike\b(?!.*cycle|.*exercise|.*spin|.*motor\s*)', 'Motorcycle'),
            (r'\bmotor\s*bike\b|\bmotor\s*cycle\b', 'Motorcycle'),
            (r'\bcar\s*sun\s*shade\b|\bwindshield\s*shade\b', 'Car Sun Shade'),
            (r'\bcar\s*organizer\b|\bauto\s*organizer\b', 'Car Organizer'),
        ]
        for pattern, product_type in auto_patterns:
            if re.search(pattern, product_lower, re.IGNORECASE):
                return product_type, self._get_context_keywords(product_name, product_type)

        baby_patterns = [
            (r'\bbaby\s*formula\b|\binfant\s*formula\b|\bbaby\s*milk\b', 'Baby Formula'),
            (r'\bbaby\s*food\b|\bpuree\b|\binfant\s*food\b', 'Baby Food'),
            (r'\bbaby\s*oil\b|\binfant\s*oil\b', 'Baby Oil'),
            (r'\bbaby\s*lotion\b|\binfant\s*lotion\b', 'Baby Lotion'),
            (r'\bbaby\s*powder\b|\btalcum\s*powder\b(?!.*women|.*men)', 'Baby Powder'),
            (r'\bbaby\s*soap\b|\binfant\s*soap\b|\bbaby\s*bath\b', 'Baby Soap'),
            (r'\bbaby\s*shampoo\b|\binfant\s*shampoo\b', 'Baby Shampoo'),
            (r'\bbaby\s*monitor\b|\binfant\s*monitor\b', 'Baby Monitor'),
            (r'\bbaby\s*bouncer\b|\binfant\s*bouncer\b|\bkids\s*bouncer\b', 'Baby Bouncer'),
            (r'\bbaby\s*swing\b|\binfant\s*swing\b', 'Baby Swing'),
            (r'\bbaby\s*chair\b|\bhigh\s*chair\b|\bfeeding\s*chair\b', 'High Chair'),
            (r'\bbaby\s*crib\b|\bbaby\s*cot\b|\bbaby\s*bed\b|\bnursery\s*cot\b', 'Baby Crib'),
            (r'\bbaby\s*walker\b|\binfant\s*walker\b|\bwalking\s*ring\b', 'Baby Walker'),
            (r'\bbaby\s*rocker\b|\binfant\s*rocker\b', 'Baby Rocker'),
            (r'\bkids\s*scooter\b|\bchildren\s*scooter\b|\bbaby\s*scooter\b', 'Kids Scooter'),
            (r'\bbaby\s*teether\b|\binfant\s*teether\b', 'Teether'),
            (r'\bbaby\s*rattle\b|\binfant\s*rattle\b', 'Baby Rattle'),
            (r'\bbaby\s*blanket\b|\bswaddle\b|\binfant\s*blanket\b', 'Baby Blanket'),
        ]
        for pattern, product_type in baby_patterns:
            if re.search(pattern, product_lower, re.IGNORECASE):
                return product_type, self._get_context_keywords(product_name, product_type)

        phone_patterns = [
            (r'\bsmartphone\b|\bandroid\s*phone\b|\bmobile\s*phone\b|\bcell\s*phone\b', 'Smartphone'),
            (r'\biphone\s*\d+\b|\bapple\s*phone\b', 'iPhone'),
            (r'\bphone\s*case\b|\bphone\s*cover\b|\bback\s*cover\b(?!.*sofa|.*seat)', 'Phone Case'),
            (r'\btablet\s*case\b|\bipad\s*case\b', 'Tablet Case'),
            (r'\bscreen\s*protector\b|\btempered\s*glass\b|\bscreen\s*guard\b', 'Screen Protector'),
            (r'\bpower\s*bank\b|\bpowerbank\b|\bportable\s*charger\b', 'Power Bank'),
            (r'\busb\s*c\s*charger\b|\biphone\s*charger\b|\b20w\s*charger\b|\bcharger\s*plug\b|\bcharging\s*plug\b', 'Phone Charger'),
            (r'\bcharging\s*cable\b|\bdata\s*cable\b|\btype\s*c\s*cable\b|\blightning\s*cable\b', 'Charging Cable'),
            (r'\bphone\s*holder\b|\bcar\s*phone\s*holder\b|\bphone\s*mount\b', 'Phone Holder'),
            (r'\bsim\s*card\b|\bsim\s*tray\b', 'SIM Card'),
            (r'\bphone\s*strap\b|\bphone\s*lanyard\b', 'Phone Strap'),
            (r'\breplacement\s*screen\b|\blcd\s*screen\b(?!.*tv)', 'Phone Screen'),
            (r'\bwireless\s*charger\b|\bqi\s*charger\b|\binductive\s*charger\b', 'Wireless Charger'),
            (r'\bcordless\s*phone\b|\blandline\s*phone\b|\btelefone\b', 'Landline Phone'),
        ]
        for pattern, product_type in phone_patterns:
            if re.search(pattern, product_lower, re.IGNORECASE):
                return product_type, self._get_context_keywords(product_name, product_type)

        computing_patterns = [
            (r'\blaptop\b|\bnotebook\s*computer\b|\bchromebook\b', 'Laptop'),
            (r'\bdesktop\s*computer\b|\bdesktop\s*pc\b|\ball.?in.?one\s*computer\b', 'Desktop Computer'),
            (r'\bmonitor\b(?!.*blood|.*health)', 'Computer Monitor'),
            (r'\bkeyboard\b(?!.*piano|.*music|.*instrument|.*midi|.*gaming)', 'Keyboard'),
            (r'\bcomputer\s*mouse\b|\bwireless\s*mouse\b|\boptical\s*mouse\b', 'Computer Mouse'),
            (r'\blaptop\s*bag\b|\blaptop\s*sleeve\b|\blaptop\s*backpack\b', 'Laptop Bag'),
            (r'\bexternal\s*hard\s*drive\b|\bexternal\s*hdd\b|\bportable\s*hard\s*drive\b', 'External Hard Drive'),
            (r'\bflash\s*drive\b|\busb\s*drive\b|\bpen\s*drive\b|\bpendrive\b|\bmemory\s*stick\b', 'USB Flash Drive'),
            (r'\bmemory\s*card\b|\bsd\s*card\b|\bmicro\s*sd\b', 'Memory Card'),
            (r'\bram\s*memory\b|\bram\b(?!.*goat)', 'RAM'),
            (r'\bssd\b|\bsolid\s*state\s*drive\b', 'SSD'),
            (r'\bnetwork\s*switch\b|\bethernet\s*switch\b|\bhub\b(?!.*cap)', 'Network Switch'),
            (r'\bwifi\s*router\b|\bwireless\s*router\b|\binternet\s*router\b', 'Router'),
            (r'\bups\s*battery\b|\buninterruptible\s*power\b', 'UPS Battery Backup'),
            (r'\bcooling\s*pad\b|\blaptop\s*cooler\b|\bcooling\s*fan\b(?!.*tower)', 'Laptop Cooling Pad'),
            (r'\bhdmi\s*cable\b|\bvga\s*cable\b|\bdisplayport\b', 'HDMI Cable'),
            (r'\bhub\s*usb\b|\busb\s*hub\b', 'USB Hub'),
            (r'\bdocking\s*station\b|\blaptop\s*dock\b', 'Docking Station'),
            (r'\bwebcam\b|\bweb\s*camera\b(?!.*security)', 'Webcam'),
            (r'\bheadset\b(?!.*gaming|.*bluetooth|.*wireless)', 'Computer Headset'),
            (r'\bcomputer\s*speaker\b|\bdesk\s*speaker\b', 'Computer Speaker'),
        ]
        for pattern, product_type in computing_patterns:
            if re.search(pattern, product_lower, re.IGNORECASE):
                return product_type, self._get_context_keywords(product_name, product_type)

        product_patterns = [
            (r'\brefrigerator\b|\bfridge\b|\bmini\s*fridge\b|\bdoor\s*fridge\b|\bbar\s*fridge\b', 'Refrigerator'),
            (r'\bair\s*conditioner\b|\bsplit\s*ac\b|\bwindow\s*ac\b|\bportable\s*ac\b|\bac\s*unit\b|\bhvac\b', 'Air Conditioner'),
            (r'\bwashing\s*machine\b|\bwasher\s*dryer\b|\bsemi\s*automatic\s*washing\b|\bfully\s*automatic\s*washing\b', 'Washing Machine'),
            (r'\bmicrowave\s*oven\b|\bmicrowave\b', 'Microwave Oven'),
            (r'\belectric\s*kettle\b|\bkettle\b|\bhot\s*pot\b', 'Kettle'),
            (r'\bfood\s*processor\b|\bprocessor\b', 'Food Processor'),
            (r'\binduction\s*cooker\b|\binduction\s*cooktop\b|\binduction\s*stove\b', 'Induction Cooker'),
            (r'\brice\s*cooker\b|\brice\s*steamer\b', 'Rice Cooker'),
            (r'\bpressure\s*cooker\b', 'Pressure Cooker'),
            (r'\btoaster\b|\btoaster\s*oven\b', 'Toaster'),
            (r'\bcoffee\s*maker\b|\bcoffee\s*machine\b|\bcoffeemaker\b|\bcoffeemachine\b', 'Coffee Maker'),
            (r'\bblender\b|\bmixer\b|\bstand\s*mixer\b|\bhand\s*mixer\b', 'Blender'),
            (r'\bjuicer\b|\bjuice\s*extractor\b', 'Juicer'),
            (r'\bair\s*fryer\b|\bairfryer\b|\bfry\s*d\s*air\b', 'Air Fryer'),
            (r'\belectric\s*cooker\b|\bcooker\b(?!.*pressure)', 'Electric Cooker'),
            (r'\bstand\s*fan\b|\bceiling\s*fan\b|\bexhaust\s*fan\b|\bventilating\s*fan\b', 'Fan'),
            (r'\bair\s*purifier\b|\bair\s*cleaner\b', 'Air Purifier'),
            (r'\bdehumidifier\b', 'Dehumidifier'),
            (r'\bhumidifier\b', 'Humidifier'),
            
            (r'\bchopping\s*board\b|\bcutting\s*board\b|\bboard\b(?!.*game)', 'Chopping Board'),
            (r'\bvegetable\s*chopper\b|\bchopper\b|\bfood\s*chopper\b|\bmanual\s*chopper\b|\bmulti.*chopper\b', 'Food Chopper'),
            (r'\bvegetable\s*slicer\b|\bfood\s*slicer\b|\bslicer\b|\bvegetable\s*cutter\b|\bfood\s*cutter\b|\bfrench\s*fry\s*cutter\b|\bdicer\b', 'Food Chopper'),
            (r'\bpeeler\b|\bvegetable\s*peeler\b', 'Peeler'),
            (r'\bgrater\b|\bcheese\s*grater\b', 'Grater'),
            (r'\bwhisk\b|\bhand\s*whisk\b|\bballoon\s*whisk\b', 'Whisk'),
            (r'\bspatula\b|\bturner\b|\bflipper\b', 'Spatula'),
            (r'\bcolander\b|\bstrainer\b|\bsieve\b', 'Colander'),
            (r'\bmeasuring\s*cup\b|\bmeasuring\s*jug\b', 'Measuring Cup'),
            (r'\brolling\s*pin\b', 'Rolling Pin'),
            (r'\bice\s*cube\s*tray\b|\bice\s*tray\b', 'Ice Cube Tray'),
            
            (r'\bnon\s*stick\s*cookware\b|\bnonstick\s*cookware\b', 'Cookware Set'),
            (r'\bfry\s*pan\b|\bfrying\s*pan\b|\bskillet\b', 'Frying Pan'),
            (r'\bnon\s*stick\s*fry\s*pan\b|\bnonstick\s*fry\s*pan\b', 'Non-Stick Frying Pan'),
            (r'\bgriddle\b|\bflat\s*griddle\b', 'Griddle'),
            (r'\bstockpot\b|\bstock\s*pot\b', 'Stockpot'),
            (r'\bpot\s*set\b|\bpots\s*set\b|\bpot\b.*\bset\b|\bcookware\s*set\b', 'Pot Set'),
            (r'\bsauce\s*pan\b|\bsaucepan\b', 'Saucepan'),
            (r'\bnon\s*stick\s*pots\b|\bnonstick\s*pots\b|\bnon\s*stick\s*pot\b|\bnonstick\s*pot\b', 'Non-Stick Pots'),
            (r'\bgranite\s*pots\b|\bgranite\s*pot\b|\bdiecast\s*pots\b|\bcast\s*pots\b|\bcast\s*aluminum\s*pot\b', 'Cookware Pots'),
            (r'\bdeep\s*fry\s*pan\b|\bdeep\s*fry\s*pot\b', 'Deep Fry Pan'),
            (r'\bsteamer\s*pot\b|\bsteamer\s*basket\b', 'Steamer'),
            (r'\bpressure\s*cooker\b', 'Pressure Cooker'),
            (r'\bkitchen\s*pot\b|\bcooking\s*pot\b|\bcooking\s*pots\b|\bquality\s*pot\b|\bquality\s*pots\b', 'Pot'),
            
            (r'\bdinner\s*set\b|\bdinnerware\s*set\b', 'Dinnerware Set'),
            (r'\bdining\s*set\b', 'Dining Set'),
            (r'\bplate\s*set\b|\bplates\b(?!.*wall)', 'Plate Set'),
            (r'\bbowl\s*set\b|\bbowls\b(?!.*wall)', 'Bowl Set'),
            (r'\bcoffee\s*cup\b|\btea\s*cup\b|\bcup\b(?!.*holder)', 'Coffee Cup'),
            (r'\bmug\b|\bcoffee\s*mug\b', 'Coffee Mug'),
            (r'\bsaucer\b', 'Saucer'),
            
            (r'\bface\s*cream\b|\bface\s*moisturizer\b|\bday\s*cream\b|\bnight\s*cream\b', 'Face Cream'),
            (r'\bbody\s*lotion\b|\bbody\s*moisturizer\b', 'Body Lotion'),
            (r'\bserum\b|\bface\s*serum\b', 'Face Serum'),
            (r'\btoner\b|\bface\s*toner\b', 'Face Toner'),
            (r'\bcleanser\b|\bface\s*wash\b|\bface\s*cleanser\b', 'Face Cleanser'),
            (r'\bsunscreen\b|\bsun\s*screen\b|\bspf\b', 'Sunscreen'),
            (r'\bmask\b|\bface\s*mask\b|\bsheet\s*mask\b', 'Face Mask'),
            (r'\bexfoliator\b|\bface\s*scrub\b', 'Face Scrub'),
            (r'\bhair\s*oil\b|\bhair\s*serum\b', 'Hair Oil'),
            (r'\bshampoo\b|\bconditioner\b', 'Shampoo'),
            (r'\bsoap\b|\bbody\s*soap\b', 'Soap'),
            (r'\btoothpaste\b|\btooth\s*paste\b', 'Toothpaste'),
            (r'\btoothbrush\b|\belectric\s*toothbrush\b', 'Toothbrush'),
            (r'\bgummi\w*\b|\bbiotin\b|\bvitamin\w*\b|\bcalcium\b|\bmultivitamin\b', 'Vitamins'),
            (r'\bhip\s*enlargement\b|\bhair\s*skin\s*nail\b', 'Vitamins'),
            (r'\bbonnet\b|\bsleep\s*cap\b|\bhair\s*cap\b', 'Hair Accessories'),
            
            (r'\btelevision\b|\bsmart\s*tv\b|\bled\s*tv\b|\boled\s*tv\b|\blcd\s*tv\b|\b4k\s*tv\b|\buhd\s*tv\b|\btv\b(?!.*stand)', 'Television'),
            (r'\bremote\s*control\b|\bremote\b(?!.*car)', 'Remote Control'),
            (r'\bheadphone\b|\bheadphones\b(?!.*fitness|.*tracker)', 'Headphones'),
            (r'\bwireless\s*earbud\b|\blisten\s*ear\b|\bbluetooth\s*earbud\b|\bfreepods\b|\btune\s*510\b', 'Headphones'),
            (r'\bspeaker\b|\bbluetooth\s*speaker\b|\bportable\s*speaker\b', 'Speaker'),
            (r'\bcamera\b|\bdslr\s*camera\b|\bmirrorless\s*camera\b|\bpoint\s*and\s*shoot\b', 'Camera'),
            (r'\bwebcam\b|\bweb\s*camera\b', 'Webcam'),
            (r'\btripod\b', 'Tripod'),
            (r'\bring\s*light\b', 'Ring Light'),
            (r'\bpower\s*bank\b|\bportable\s*charger\b', 'Power Bank'),
            (r'\bcharger\b|\bphone\s*charger\b|\bwall\s*charger\b|\btravel\s*charger\b|\b20w\s*charger\b|\busb\s*c\s*charger\b', 'Phone Charger'),
            (r'\bdata\s*cable\b|\bcharging\s*cable\b|\busb\s*cable\b', 'USB Cable'),
            (r'\badapter\b(?!.*phone|.*charger)|\busb\s*adapter\b|\bpower\s*adapter\b(?!.*phone)', 'USB Adapter'),
            (r'\bphone\s*case\b|\bphone\s*cover\b|\bcase\b.*\bphone\b|\bgalaxy\s*(z\s*flip|z\s*fold|s\d+)\s*case\b|\biphone\s*\d+\s*pro\s*case\b', 'Phone Case'),
            (r'\btablet\s*case\b|\bipad\s*case\b|\bgalaxy\s*tab\s*case\b', 'Tablet Case'),
            (r'\blaptop\s*stand\b|\blaptop\s*holder\b', 'Laptop Stand'),
            (r'\bmonitor\s*stand\b|\bmonitor\s*arm\b', 'Monitor Stand'),
            (r'\bkeyboard\b|\bwireless\s*keyboard\b|\bbluetooth\s*keyboard\b', 'Keyboard'),
            (r'\bmouse\b|\bwireless\s*mouse\b|\bbluetooth\s*mouse\b', 'Mouse'),
            (r'\brouter\b|\bwifi\s*router\b|\bwireless\s*router\b', 'Router'),
            
            (r'\boffice\s*chair\b|\bdesk\s*chair\b|\bcomputer\s*chair\b', 'Office Chair'),
            (r'\bdesk\b(?!.*lamp)', 'Desk'),
            (r'\bfiling\s*cabinet\b|\bfile\s*cabinet\b', 'Filing Cabinet'),
            (r'\bbookshelf\b|\bbookcase\b', 'Bookshelf'),
            (r'\bdesk\s*lamp\b|\btable\s*lamp\b|\blamp\b(?!.*uv)', 'Desk Lamp'),
            (r'\bdesk\s*organizer\b|\boffice\s*organizer\b', 'Desk Organizer'),
            
            (r'\bpillow\b(?!.*case)', 'Pillow'),
            (r'\bpillowcase\b|\bpillow\s*case\b(?!.*baby|.*nursery|.*toddler)', 'Pillowcase'),
            (r'\bsheet\s*set\b|\bbed\s*sheet\b|\bsheets\b', 'Bed Sheet Set'),
            (r'\bquilt\b|\bcomforter\b|\bduvet\b', 'Quilt'),
            (r'\bblanket\b|\bthrow\s*blanket\b', 'Blanket'),
            (r'\bmattress\s*protector\b|\bmattress\s*pad\b', 'Mattress Protector'),
            
            (r'\bcarpet\b|\barea\s*rug\b|\bliving\s*room\s*carpet\b', 'Carpet'),
            (r'\bdoormat\b|\bwelcome\s*mat\b|\bentrance\s*mat\b', 'Doormat'),
            (r'\bbath\s*mat\b|\bbathroom\s*mat\b', 'Bath Mat'),
            
            (r'\bdetergent\b|\blaundry\s*detergent\b', 'Laundry Detergent'),
            (r'\bsoftener\b|\bfabric\s*softener\b', 'Fabric Softener'),
            (r'\bstain\s*remover\b', 'Stain Remover'),
            (r'\banti\s*vibration\b|\banti\s*vibration\s*pad\b|\bvibration\s*pad\b|\bwashing\s*machine\s*pad\b', 'Washing Machine Accessory'),
            (r'\bgripper\b|\bcarpet\s*gripper\b|\bmattress\s*gripper\b|\breusable\s*gripper\b', 'Laundry Accessory'),
            
            (r'\bdiaper\b|\bnappy\b', 'Diapers'),
            (r'\bdiaper\s*bag\b|\blunch\s*bag\b', 'Diaper Bag'),
            (r'\bbaby\s*wipe\b|\bwipe\b(?!.*screen)', 'Baby Wipes'),
            (r'\bbaby\s*bottle\b|\bnursing\s*bottle\b', 'Baby Bottle'),
            (r'\bbaby\s*stroller\b|\bpram\b', 'Baby Stroller'),
            (r'\bbaby\s*carrier\b|\bbaby\s*wrap\b', 'Baby Carrier'),
            (r'\bcrib\s*mattress\b|\bbaby\s*mattress\b', 'Crib Mattress'),
            
            (r'\bbuilding\s*block\b|\blego\b(?!.*movie)', 'Building Blocks'),
            (r'\bdoll\b|\bdolls\b', 'Doll'),
            (r'\baction\s*figure\b', 'Action Figure'),
            (r'\bboard\s*game\b', 'Board Game'),
            (r'\bpuzzle\b', 'Puzzle'),
            (r'\bstuffed\s*animal\b|\bplush\s*toy\b|\bteddy\s*bear\b|\bplushie\b', 'Plush Toy'),
            (r'\bremote\s*control\s*(car|toy|truck|drone)\b|\brc\s*(car|toy|truck|drone)\b', 'RC Toy'),
            (r'\btoy\s*car\b|\bdie.?cast\s*car\b(?!.*cookware)', 'Toy Car'),
            (r'\bslime\b|\bkinetic\s*sand\b', 'Slime Toy'),
            (r'\bwater\s*gun\b|\bnerf\b|\bsquirt\s*gun\b', 'Water Gun'),
            (r'\bjigsaw\b', 'Jigsaw Puzzle'),
            (r'\bread\s*light\s*therapy\b|\bblue\s*light\s*therapy\b', 'Light Therapy Device'),
            
            (r'\byoga\s*mat\b|\bexercise\s*mat\b', 'Yoga Mat'),
            (r'\bdumbbell\b|\badjustable\s*dumbbell\b', 'Dumbbell'),
            (r'\bbarbell\b|\bweight\s*bar\b', 'Barbell'),
            (r'\btreadmill\b', 'Treadmill'),
            (r'\bexercise\s*bike\b|\bspin\s*bike\b|\bstationary\s*bike\b', 'Exercise Bike'),
            (r'\bjump\s*rope\b|\bskipping\s*rope\b', 'Jump Rope'),
            (r'\bpull.?up\s*bar\b|\bchin.?up\s*bar\b', 'Pull-Up Bar'),
            (r'\bab\s*roller\b|\bab\s*wheel\b', 'Ab Roller'),
            (r'\bresistance\s*band\b|\bexercise\s*band\b', 'Resistance Band'),
            (r'\bkettle\s*bell\b|\bkettlebell\b', 'Kettlebell'),
            (r'\bweightlifting\s*glove\b|\bgym\s*glove\b', 'Gym Gloves'),
            (r'\bsoccer\s*ball\b|\bfootball\b(?!.*jersey|.*shirt)', 'Football'),
            (r'\bbasketball\b(?!.*jersey|.*shirt|.*hoop\s*stand)', 'Basketball'),
            (r'\bvolleyball\b', 'Volleyball'),
            (r'\btennis\s*racket\b|\btennis\s*racquet\b', 'Tennis Racket'),
            (r'\bbadminton\s*racket\b|\bbadminton\s*set\b', 'Badminton Racket'),
            (r'\btable\s*tennis\b|\bping\s*pong\b', 'Table Tennis'),
            (r'\bboxing\s*glove\b|\bboxing\s*bag\b', 'Boxing Equipment'),
            (r'\bswimming\s*goggle\b|\bswim\s*goggle\b', 'Swimming Goggles'),
            (r'\bswimwear\b|\bbikini\b|\bswimsuit\b', 'Swimwear'),
            (r'\bcycling\s*helmet\b|\bbike\s*helmet\b|\bbicycle\s*helmet\b', 'Cycling Helmet'),
            (r'\bbicycle\b|\bcycling\s*bike\b', 'Bicycle'),
            (r'\bskateboard\b|\bscooter\b', 'Skateboard'),
            (r'\bgolf\s*club\b|\bgolf\s*set\b|\bgolf\s*ball\b', 'Golf Equipment'),
            (r'\bfishing\s*rod\b|\bfishing\s*reel\b|\bfishing\s*line\b', 'Fishing Equipment'),
            (r'\bcamping\s*tent\b|\bbackpacking\s*tent\b', 'Camping Tent'),
            (r'\bsleeping\s*bag\b', 'Sleeping Bag'),
            (r'\bhiking\s*bag\b|\bcamping\s*bag\b', 'Hiking Bag'),
            (r'\bkneepads?\b|\bknee\s*pad\b|\bknee\s*brace\b|\bknee\s*support\b', 'Knee Support'),
            (r'\bankle\s*support\b|\bwrist\s*support\b|\belbow\s*support\b', 'Sports Support'),
            (r'\bsports\s*jersey\b|\bfootball\s*jersey\b|\bbasketball\s*jersey\b', 'Sports Jersey'),
            
            (r'\brunning\s*shoe\b|\bsneaker\b|\btraining\s*shoe\b', 'Running Shoes'),
            (r'\bcasual\s*shoe\b|\bcasual\s*sneaker\b', 'Casual Shoes'),
            (r'\bboot\b|\bhiking\s*boot\b|\bwork\s*boot\b', 'Boots'),
            (r'\bsandal\b|\bflip\s*flop\b|\bslippers?\b', 'Sandals'),
            (r'\bwedge\s*heel\b|\bstiletto\b|\bhigh\s*heel\b', 'High Heels'),
            (r'\binfant\s*shoe\b|\bbaby\s*shoe\b|\bkid\s*shoe\b|\bchildren\s*shoe\b', 'Kids Shoes'),
            
            (r'\bhandbag\b|\bwomen\s*handbag\b|\bladies\s*handbag\b', 'Handbag'),
            (r'\bbackpack\b|\bschool\s*backpack\b|\btravel\s*backpack\b', 'Backpack'),
            (r'\btravel\s*bag\b|\bluggage\b|\bsuitcase\b|\btrolley\s*bag\b', 'Travel Bag'),
            (r'\bcrossbody\s*bag\b|\bcross\s*body\s*bag\b', 'Crossbody Bag'),
            (r'\btote\s*bag\b', 'Tote Bag'),
            (r'\bclutch\s*bag\b|\bevening\s*clutch\b', 'Clutch Bag'),
            (r'\bgym\s*bag\b|\bduffle\s*bag\b|\bduffel\s*bag\b', 'Gym Bag'),
            (r'\bbriefcase\b', 'Briefcase'),
            (r'\bwallet\b|\bpurse\b(?!.*lips)', 'Wallet'),
            
            (r'\bsmart\s*watch\b|\bfitness\s*watch\b|\bfitness\s*tracker\b', 'Smart Watch'),
            (r'\bwatch\b|\bwrist\s*watch\b', 'Watch'),
            (r'\bbracelet\b|\bwrist\s*bracelet\b|\banka\b(?!.*wrist)', 'Bracelet'),
            (r'\bnecklace\b|\bpendant\b|\bchain\s*necklace\b', 'Necklace'),
            (r'\bearring\b|\bearrings\b|\bstud\s*earring\b', 'Earrings'),
            (r'\bring\b(?!.*light|.*guard|.*notebook)', 'Ring'),
            (r'\bsunglasses?\b|\bshades\b', 'Sunglasses'),
            (r'\bbangle\b', 'Bangle'),
            
            (r'\bacoustic\s*guitar\b|\belectric\s*guitar\b|\bguitar\b', 'Guitar'),
            (r'\bpiano\b|\bkeyboard\s*piano\b|\bdigital\s*piano\b', 'Piano'),
            (r'\bkeyboard\s*instrument\b|\bmusic\s*keyboard\b', 'Music Keyboard'),
            (r'\bdrum\s*set\b|\bdrum\s*kit\b|\bdrums?\b(?!.*machine)', 'Drum Set'),
            (r'\bviolin\b|\bfiddle\b', 'Violin'),
            (r'\bflute\b|\brecorder\s*instrument\b', 'Flute'),
            (r'\btrumpet\b|\btrombone\b|\bsaxophone\b', 'Brass Instrument'),
            (r'\bmicrophone\b|\bmic\b(?!.*-)', 'Microphone'),
            (r'\bamplifier\b|\bguitar\s*amp\b', 'Amplifier'),
            (r'\bukulele\b', 'Ukulele'),
            
            (r'\bplaystation\b|\bps5\b|\bps4\b|\bps3\b|\bpsp\b', 'PlayStation'),
            (r'\bxbox\b', 'Xbox'),
            (r'\bnintendo\s*switch\b|\bnintendo\b(?!.*game)', 'Nintendo'),
            (r'\bgaming\s*console\b|\bvideo\s*game\s*console\b', 'Gaming Console'),
            (r'\bgaming\s*controller\b|\bgame\s*controller\b|\bjoystick\b', 'Game Controller'),
            (r'\bgaming\s*keyboard\b', 'Gaming Keyboard'),
            (r'\bgaming\s*mouse\b', 'Gaming Mouse'),
            (r'\bgaming\s*headset\b', 'Gaming Headset'),
            (r'\bgaming\s*chair\b', 'Gaming Chair'),
            (r'\bvideo\s*game\b|\bpc\s*game\b', 'Video Game'),
            
            (r'\bdog\s*food\b|\bpet\s*food\b|\bdog\s*treat\b', 'Dog Food'),
            (r'\bcat\s*food\b|\bcat\s*treat\b', 'Cat Food'),
            (r'\bdog\s*collar\b|\bpet\s*collar\b', 'Dog Collar'),
            (r'\bdog\s*leash\b|\bpet\s*leash\b', 'Dog Leash'),
            (r'\bpet\s*bed\b|\bdog\s*bed\b|\bcat\s*bed\b', 'Pet Bed'),
            (r'\bcat\s*litter\b|\bpet\s*litter\b', 'Cat Litter'),
            (r'\baquarium\b|\bfish\s*tank\b', 'Aquarium'),
            (r'\bbird\s*cage\b|\bpet\s*cage\b', 'Bird Cage'),
            (r'\bpet\s*shampoo\b|\bdog\s*shampoo\b', 'Pet Shampoo'),
            (r'\bpet\s*toy\b|\bdog\s*toy\b|\bcat\s*toy\b', 'Pet Toy'),
            
            (r'\bgarden\s*hose\b|\bwater\s*hose\b', 'Garden Hose'),
            (r'\blawn\s*mower\b|\bgrass\s*cutter\b', 'Lawn Mower'),
            (r'\bplant\s*pot\b|\bflower\s*pot\b|\bplanter\b', 'Plant Pot'),
            (r'\bwatering\s*can\b|\bgarden\s*sprayer\b', 'Watering Can'),
            (r'\bgarden\s*tool\b|\bgarden\s*fork\b|\bgarden\s*shovel\b', 'Garden Tool'),
            (r'\bgarden\s*chair\b|\bpatio\s*chair\b|\boutdoor\s*chair\b', 'Outdoor Chair'),
            (r'\bgarden\s*table\b|\bpatio\s*table\b|\boutdoor\s*table\b', 'Outdoor Table'),
            (r'\bbarbeque\b|\bbbq\s*grill\b|\bcharcoal\s*grill\b', 'BBQ Grill'),
            (r'\bgenerator\b(?!.*app)', 'Generator'),
            (r'\bpool\b(?!.*billiard|.*snooker)', 'Swimming Pool'),
            (r'\bhammock\b', 'Hammock'),
            (r'\btent\b(?!.*camping)', 'Tent'),
            (r'\bgazebo\b|\bpergola\b', 'Gazebo'),
            (r'\bcompost\b|\bfertilizer\b', 'Fertilizer'),
            (r'\binsecticide\b|\bpesticide\b|\bbug\s*spray\b', 'Pesticide'),
            
            (r'\bdetergent\s*powder\b|\bwashing\s*powder\b|\blaundry\s*powder\b', 'Laundry Powder'),
            (r'\bdish\s*wash\b|\bdishwashing\s*liquid\b|\bdishing\s*liquid\b', 'Dishwashing Liquid'),
            (r'\bfloor\s*cleaner\b|\bmop\s*cleaner\b', 'Floor Cleaner'),
            (r'\btoilet\s*cleaner\b|\bbathroom\s*cleaner\b', 'Toilet Cleaner'),
            (r'\bair\s*freshener\b|\broom\s*freshener\b|\bcar\s*freshener\b', 'Air Freshener'),
            (r'\bscour\s*pad\b|\bscouring\s*pad\b|\bscrub\s*pad\b|\bscrubber\b', 'Scouring Pad'),
            (r'\btrash\s*bag\b|\bgarbage\s*bag\b', 'Garbage Bag'),
            (r'\bdisposable\s*glove\b|\brubber\s*glove\b|\bkitchen\s*glove\b', 'Cleaning Gloves'),
            (r'\bcigar\b|\bcigarette\b', 'Tobacco'),
            (r'\bsnack\b|\bchip\b(?!.*chopper|.*cutter)', 'Snack'),
            (r'\brice\b(?!.*cooker)', 'Rice'),
            (r'\bflour\b|\bcooking\s*oil\b|\bvegetable\s*oil\b|\bpalm\s*oil\b', 'Cooking Ingredient'),
            
            (r'\bnovel\b|\bfiction\s*book\b|\bstory\s*book\b', 'Novel'),
            (r'\bbible\b|\bquran\b|\bholy\s*book\b|\breligious\s*book\b', 'Religious Book'),
            (r'\bself.?help\s*book\b|\bmotivational\s*book\b', 'Self Help Book'),
            (r'\btextbook\b|\beducational\s*book\b|\bacademic\s*book\b', 'Textbook'),
            (r'\bchildren\s*book\b|\bkids\s*book\b|\bpicture\s*book\b', "Children's Book"),
            (r'\bdvd\b|\bblueray\b|\bblu.?ray\b', 'DVD'),
            (r'\bvcd\b(?!.*player)', 'DVD'),
            (r'\bnotebook\b|\bexercise\s*book\b|\bjotter\b', 'Notebook'),
            (r'\bpen\b(?!.*drive|.*dant)', 'Pen'),
            (r'\bpencil\b', 'Pencil'),
            (r'\bcrayo\b', 'Crayon'),
            (r'\bmarker\b|\bhighlighter\b', 'Marker'),
            (r'\bfile\s*folder\b|\bdocument\s*file\b', 'File Folder'),
            (r'\bcalculator\b', 'Calculator'),
            
            (r'\bsoldering\s*iron\b|\bsoldering\s*kit\b', 'Soldering Iron'),
            (r'\bmultimeter\b|\bvoltmeter\b', 'Multimeter'),
            (r'\bcable\s*tie\b|\bzip\s*tie\b', 'Cable Tie'),
            (r'\bsafety\s*vest\b|\bsafety\s*jacket\b|\breflective\s*vest\b', 'Safety Vest'),
            (r'\bhard\s*hat\b|\bsafety\s*helmet\b', 'Safety Helmet'),
            (r'\bsafety\s*glove\b|\bwork\s*glove\b', 'Work Gloves'),
            (r'\bsafety\s*boot\b|\bsteel\s*toe\s*boot\b', 'Safety Boots'),
            (r'\bfire\s*extinguisher\b', 'Fire Extinguisher'),
            (r'\bfirst\s*aid\b', 'First Aid'),
            (r'\bduct\s*tape\b|\bmasking\s*tape\b', 'Industrial Tape'),
            (r'\blevel\b(?!.*vitamin|.*supplement|.*health)|\bspirit\s*level\b', 'Spirit Level'),
            (r'\bmeasuring\s*tape\b|\btape\s*measure\b', 'Measuring Tape'),
            (r'\bhand\s*drill\b|\belectric\s*drill\b|\bcordless\s*drill\b', 'Power Drill'),
            (r'\bscrewdriver\b|\bscrewdriver\s*set\b', 'Screwdriver'),
            (r'\bspanner\b|\bwrench\b', 'Wrench'),
            (r'\bhammer\b(?!.*drill)', 'Hammer'),
            (r'\bsaw\b(?!.*seesaw)', 'Saw'),
            (r'\bpliers\b', 'Pliers'),
            (r'\btool\s*box\b|\btool\s*kit\b|\btool\s*set\b(?!.*kitchen)', 'Tool Set'),
            (r'\b3d\s*printer\b', '3D Printer'),
            (r'\blab\s*coat\b|\bwhite\s*coat\b', 'Lab Coat'),
            (r'\bbeaker\b|\btest\s*tube\b|\bpipette\b', 'Lab Equipment'),
            
            (r'\bsmartphone\b|\bandroid\s*phone\b|\bios\s*phone\b', 'Smartphone'),
            (r'\bsim\s*card\b', 'SIM Card'),
            (r'\bipad\b', 'iPad'),
            (r'\bscreenguard\b|\bscreen\s*protector\b|\btempered\s*glass\b', 'Screen Protector'),
            (r'\bphone\s*holder\b|\bcar\s*phone\s*mount\b|\bphone\s*stand\b', 'Phone Holder'),
            
            (r'\blaptop\b|\bnotebook\s*computer\b', 'Laptop'),
            (r'\bdesktop\s*computer\b|\bdesktop\s*pc\b', 'Desktop Computer'),
            (r'\bprinter\b(?!.*3d)', 'Printer'),
            (r'\bscanner\b(?!.*body)', 'Scanner'),
            (r'\bwebcam\b|\bweb\s*cam\b', 'Webcam'),
            (r'\bnetwork\s*switch\b|\bethernet\s*switch\b', 'Network Switch'),
            (r'\bups\b(?!.*tracking|.*shipping)', 'UPS Battery Backup'),
            (r'\bram\b(?!.*goat|\brams?\b(?!.*memory|\bcomputing))', 'RAM'),
            (r'\bgraphic\s*card\b|\bgpu\b|\bvideo\s*card\b', 'Graphics Card'),
            (r'\bcpu\b|\bprocessor\b(?!.*food)', 'CPU Processor'),
            
            (r'\bginseng\b|\bkorean\s*ginseng\b', 'Ginseng'),
            (r'\btea\b(?!.*clock|\bteaching)', 'Tea'),
            (r'\bcoffee\b(?!.*maker|machine)', 'Coffee'),
            (r'\bprotein\s*powder\b|\bwhey\s*protein\b', 'Protein Powder'),
            (r'\bmultivitamin\b|\bmultivitamins\b', 'Multivitamin'),
            (r'\bvitamin\s*d\b|\bvitamin\s*c\b|\bvitamin\s*b', 'Vitamin'),
            (r'\bhoney\b', 'Honey'),
            (r'\bjuice\b(?!.*extractor|.*machine)', 'Juice'),
            (r'\benergy\s*drink\b|\bsports\s*drink\b', 'Energy Drink'),
            (r'\bprotein\s*bar\b|\bsnack\s*bar\b', 'Protein Bar'),
            
            (r'\bcar\s*cover\b|\bvehicle\s*cover\b', 'Car Cover'),
            (r'\bcar\s*vacuum\b|\bportable\s*car\s*vacuum\b', 'Car Vacuum'),
            (r'\bdash\s*cam\b|\bdashboard\s*camera\b', 'Dash Cam'),
            (r'\btire\s*pressure\s*gauge\b|\bpressure\s*gauge\b', 'Tire Pressure Gauge'),
            (r'\bcar\s*charger\b|\bvehicle\s*charger\b', 'Car Charger'),
            (r'\bcar\s*seat\s*cover\b|\bcar\s*cushion\b', 'Car Seat Cover'),
            (r'\bcar\s*mat\b|\bfloor\s*mat\s*car\b', 'Car Floor Mat'),
            (r'\btire\b|\btyre\b', 'Tire'),
            (r'\bengine\s*oil\b|\bmotor\s*oil\b|\bcar\s*oil\b', 'Motor Oil'),
            (r'\bcar\s*battery\b|\bvehicle\s*battery\b', 'Car Battery'),
            (r'\bcar\s*wax\b|\bcar\s*polish\b|\bcar\s*shine\b', 'Car Polish'),
            (r'\bcar\s*jack\b|\bhydraulic\s*jack\b', 'Car Jack'),
            (r'\bparking\s*sensor\b|\bpdc\s*sensor\b', 'Parking Sensor'),
            (r'\bcar\s*stereo\b|\bcar\s*radio\b|\bhead\s*unit\b', 'Car Stereo'),
            (r'\bmotor\s*part\b|\bengine\s*part\b|\bspare\s*part\b', 'Auto Part'),
            (r'\bcar\s*light\b|\bheadlight\b|\btaillight\b|\bled\s*car\s*light\b', 'Car Light'),
        ]
        
        for pattern, product_type in product_patterns:
            if re.search(pattern, cleaned_name, re.IGNORECASE):
                return product_type, self._get_context_keywords(product_name, product_type)
        
        words = cleaned_name.split()
        product_words = [w for w in words if len(w) > 3 and w not in ['with', 'for', 'and', 'the', 'this', 'that', 'your']]
        
        return None, product_words[:5]
    
    def _get_context_keywords(self, product_name, product_type):
        """Extract context keywords that help disambiguate product type"""
        context_patterns = {
            'kitchen': ['kitchen', 'cooking', 'cook', 'food', 'vegetable', 'fruit', 'chef'],
            'bathroom': ['bath', 'shower', 'toilet', 'bathroom', 'wash'],
            'bedroom': ['bed', 'sleep', 'bedroom', 'pillow', 'blanket'],
            'living': ['living', 'room', 'hall', 'lounge', 'carpet', 'rug'],
            'outdoor': ['outdoor', 'garden', 'patio', 'yard', 'camping'],
            'portable': ['portable', 'wireless', 'cordless', 'battery', 'rechargeable'],
            'professional': ['professional', 'commercial', 'restaurant', 'hotel'],
            'baby': ['baby', 'infant', 'toddler', 'kid', 'child'],
            'fitness': ['fitness', 'gym', 'exercise', 'workout', 'training'],
            'ladies': ['ladies', 'women', 'female', 'girl', 'sexy', 'nightwear'],
        }
        
        product_lower = product_name.lower()
        contexts = []
        
        for context, keywords in context_patterns.items():
            if any(kw in product_lower for kw in keywords):
                contexts.append(context)
        
        return contexts
    
    def precompute_leaf_categories(self, categories_list):
        """Pre-compute which categories are leaf categories for performance"""
        leaf_status = {}
        categories_lower = {cat.lower().strip() for cat in categories_list}
        parent_paths = set()
        
        for cat_lower in categories_lower:
            parts = cat_lower.split('/')
            for i in range(1, len(parts)):
                parent_path = '/'.join(parts[:i]).strip()
                parent_paths.add(parent_path)
        
        for cat in categories_list:
            cat_lower = cat.lower().strip()
            leaf_status[cat] = cat_lower not in parent_paths
        
        return leaf_status
    
    def precompute_last_parts(self, categories_list):
        """Pre-compute the last part of each category path for performance"""
        last_parts = {}
        
        for cat in categories_list:
            parts = cat.split('/')
            for part in reversed(parts):
                part = part.strip()
                if part:
                    last_parts[cat] = part.lower()
                    break
            else:
                last_parts[cat] = ""
        
        return last_parts
    
    def get_last_category_part(self, category_path):
        """Extract the last/leaf part of a category path"""
        if not category_path:
            return ""
        parts = category_path.split('/')
        for part in reversed(parts):
            part = part.strip()
            if part:
                return part.lower()
        return ""
    
    def get_most_specific_category(self, product_type, context_keywords, categories_list, leaf_categories=None, last_category_parts=None):
        """Find the most specific (leaf) category that matches the product type"""
        if not product_type:
            return None
        
        if leaf_categories is None:
            leaf_categories = self.precompute_leaf_categories(categories_list)
        if last_category_parts is None:
            last_category_parts = self.precompute_last_parts(categories_list)
        
        product_lower = product_type.lower()

        product_type_routing = {
            'nightwear': 'Fashion / Women\'s Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Sets',
            'nightgown': 'Fashion / Women\'s Fashion / Clothing / Lingerie, Sleep & Lounge / Sleep & Lounge / Nightgowns & Sleepshirts',
            'women panties': 'Fashion / Women\'s Fashion / Clothing / Lingerie, Sleep & Lounge / Lingerie / Panties',
            'women bra': 'Fashion / Women\'s Fashion / Underwear & Sleepwear / Bra',
            'women lingerie': 'Fashion / Women\'s Fashion / Clothing / Lingerie, Sleep & Lounge / Lingerie',
            'women shapewear': 'Fashion / Women\'s Fashion / Underwear & Sleepwear / Shapewear & Leotards',
            'women camisole': 'Fashion / Women\'s Fashion / Clothing / Lingerie, Sleep & Lounge / Lingerie / Camisoles & Tanks',
            'women leggings': 'Fashion / Women\'s Fashion / Clothing / Leggings',
            'women jeans': 'Fashion / Women\'s Fashion / Clothing / Jeans',
            'women dress': 'Fashion / Women\'s Fashion / Clothing / Dresses',
            'women tops set': 'Fashion / Women\'s Fashion / Clothing / Tops & Tees',
            'women shorts': 'Fashion / Women\'s Fashion / Clothing / Shorts',
            'yoga mat': 'Sporting Goods / Sports & Fitness / Exercise & Fitness / Accessories',
            'dumbbell': 'Sporting Goods / Sports & Fitness / Exercise & Fitness / Strength Training Equipment / Dumbbells',
            'barbell': 'Sporting Goods / Sports & Fitness / Exercise & Fitness / Strength Training Equipment',
            'treadmill': 'Sporting Goods / Sports & Fitness / Exercise & Fitness / Cardio Training',
            'exercise bike': 'Sporting Goods / Sports & Fitness / Exercise & Fitness / Cardio Training',
            'jump rope': 'Sporting Goods / Sports & Fitness / Exercise & Fitness / Accessories',
            'pull-up bar': 'Sporting Goods / Sports & Fitness / Exercise & Fitness / Strength Training Equipment',
            'ab roller': 'Sporting Goods / Sports & Fitness / Exercise & Fitness / Strength Training Equipment / Core & Abdominal Trainers',
            'resistance band': 'Sporting Goods / Sports & Fitness / Exercise & Fitness / Strength Training Equipment',
            'kettlebell': 'Sporting Goods / Sports & Fitness / Exercise & Fitness / Strength Training Equipment / Kettlebells',
            'gym gloves': 'Sporting Goods / Sports & Fitness / Exercise & Fitness / Clothing',
            'football': 'Sporting Goods / Sports & Fitness / Team Sports / Football / Footballs',
            'basketball': 'Sporting Goods / Sports & Fitness / Team Sports / Basketball / Balls',
            'volleyball': 'Sporting Goods / Sports & Fitness / Team Sports / Volleyball',
            'tennis racket': 'Sporting Goods / Racquet Sports',
            'badminton racket': 'Sporting Goods / Racquet Sports / Badminton / Kit',
            'table tennis': 'Sporting Goods / Sports & Fitness / Team Sports / Table Tennis',
            'boxing equipment': 'Sporting Goods / Sports & Fitness / Exercise & Fitness / Strength Training Equipment',
            'swimming goggles': 'Sporting Goods / Sports & Fitness / Team Sports / Swimming / Goggles',
            'swimwear': 'Sporting Goods / Sports & Fitness / Team Sports / Swimming / Swimwear',
            'cycling helmet': 'Sporting Goods / Outdoor & Adventure / Cycling',
            'bicycle': 'Sporting Goods / Outdoor & Adventure / Cycling',
            'skateboard': 'Sporting Goods / Outdoor Recreation',
            'golf equipment': 'Sporting Goods / Sports & Fitness / Individual Sports / Golf',
            'fishing equipment': 'Sporting Goods / Outdoor Recreation',
            'camping tent': 'Sporting Goods / Outdoor Recreation',
            'sleeping bag': 'Sporting Goods / Outdoor Recreation',
            'hiking bag': 'Sporting Goods / Outdoor Recreation',
            'knee support': 'Health & Beauty / Medical Supplies & Equipment / Braces, Splints & Supports',
            'sports support': 'Health & Beauty / Medical Supplies & Equipment / Braces, Splints & Supports',
            'sports jersey': 'Sporting Goods / Sports & Fitness / Team Sports',
            'running shoes': 'Fashion / Men\'s Fashion / Shoes / Athletic',
            'casual shoes': 'Fashion / Men\'s Fashion / Shoes / Fashion Sneakers',
            'boots': 'Fashion / Men\'s Fashion / Shoes / Boots',
            'sandals': 'Fashion / Men\'s Fashion / Shoes / Sandals & Slides',
            'high heels': 'Fashion / Women\'s Fashion / Shoes / Heels',
            'kids shoes': "Fashion / Kids Fashion / Boys / Shoes",
            'handbag': 'Fashion / Women\'s Fashion / Handbags & Wallets / Handbags',
            'backpack': 'Fashion / Luggage & Travel Gear / Backpacks',
            'travel bag': 'Fashion / Luggage & Travel Gear / Travel Duffels',
            'crossbody bag': 'Fashion / Women\'s Fashion / Handbags & Wallets / Shoulder Bags',
            'tote bag': 'Fashion / Luggage & Travel Gear / Travel Totes',
            'clutch bag': 'Fashion / Women\'s Fashion / Handbags & Wallets / Handbags',
            'gym bag': 'Fashion / Luggage & Travel Gear / Gym Bags',
            'briefcase': 'Fashion / Luggage & Travel Gear / Briefcases',
            'wallet': 'Fashion / Men\'s Fashion / Accessories / Wallets',
            'smart watch': 'Phones & Tablets / Accessories / Smart Watches',
            'watch': 'Fashion / Watches & Sunglasses / Men\'s Watches',
            'bracelet': 'Fashion / Women\'s Fashion / Jewelry / Bracelets',
            'necklace': 'Fashion / Women\'s Fashion / Jewelry / Necklaces',
            'earrings': 'Fashion / Women\'s Fashion / Jewelry / Earrings',
            'ring': 'Fashion / Women\'s Fashion / Jewelry / Rings',
            'sunglasses': 'Fashion / Watches & Sunglasses / Sunglasses',
            'bangle': 'Fashion / Women\'s Fashion / Jewelry / Bracelets',
            'guitar': 'Musical Instruments / Guitars',
            'piano': 'Musical Instruments / Keyboards & MIDI / Pianos',
            'music keyboard': 'Musical Instruments / Keyboards & MIDI',
            'drum set': 'Musical Instruments / Drums & Percussion',
            'violin': 'Musical Instruments / Stringed Instruments',
            'flute': 'Musical Instruments / Wind & Woodwind Instruments',
            'brass instrument': 'Musical Instruments / Band & Orchestra',
            'microphone': 'Musical Instruments / Microphones & Accessories',
            'amplifier': 'Musical Instruments / Amplifiers & Effects',
            'ukulele': 'Musical Instruments / Ukuleles, Mandolins & Banjos',
            'playstation': 'Gaming / Playstation',
            'xbox': 'Gaming / Xbox',
            'nintendo': 'Gaming / Nintendo',
            'gaming console': 'Gaming / Other Gaming Systems',
            'game controller': 'Gaming / PC Gaming / Accessories / Controllers',
            'gaming keyboard': 'Gaming / PC Gaming / Accessories / Gaming Keyboards',
            'gaming mouse': 'Gaming / PC Gaming / Accessories / Gaming Mice',
            'gaming headset': 'Gaming / PC Gaming / Accessories',
            'gaming chair': 'Gaming / PC Gaming / Accessories',
            'video game': 'Gaming / PC Gaming / Games',
            'dog food': 'Pet Supplies / Dogs / Food',
            'cat food': 'Pet Supplies / Cats / Food',
            'dog collar': 'Pet Supplies / Dogs / Collars & Tags',
            'dog leash': 'Pet Supplies / Dogs / Leashes & Tethers',
            'pet bed': 'Pet Supplies / Dogs / Beds & Furniture',
            'cat litter': 'Pet Supplies / Cats / Litter & Housebreaking',
            'aquarium': 'Pet Supplies / Fish & Aquatic Pets / Aquarium Lights',
            'bird cage': 'Pet Supplies / Birds / Cages',
            'pet shampoo': 'Pet Supplies / Dogs / Grooming',
            'pet toy': 'Pet Supplies / Dogs / Toys',
            'garden hose': 'Garden & Outdoors / Gardening & Lawn Care',
            'lawn mower': 'Garden & Outdoors / Outdoor Power Tools',
            'plant pot': 'Garden & Outdoors / Gardening & Lawn Care',
            'watering can': 'Garden & Outdoors / Gardening & Lawn Care',
            'garden tool': 'Garden & Outdoors / Gardening & Lawn Care',
            'outdoor chair': 'Garden & Outdoors / Patio Furniture & Accessories',
            'outdoor table': 'Garden & Outdoors / Patio Furniture & Accessories',
            'bbq grill': 'Garden & Outdoors / Grills & Outdoor Cooking / Grills',
            'generator': 'Garden & Outdoors / Generators & Portable Power / Generators',
            'swimming pool': 'Garden & Outdoors / Pools, Hot Tubs & Supplies',
            'hammock': 'Garden & Outdoors / Patio Furniture & Accessories',
            'tent': 'Garden & Outdoors / Outdoor Storage',
            'gazebo': 'Garden & Outdoors / Patio Furniture & Accessories / Canopies, Gazebos & Pergolas',
            'fertilizer': 'Garden & Outdoors / Farm & Ranch',
            'pesticide': 'Garden & Outdoors / Farm & Ranch',
            'laundry powder': 'Grocery / Household Cleaning',
            'dishwashing liquid': 'Grocery / Dishwashing / Scouring Pads',
            'floor cleaner': 'Grocery / Household Cleaning',
            'toilet cleaner': 'Grocery / Household Cleaning',
            'air freshener': 'Grocery / Air Fresheners',
            'scouring pad': 'Grocery / Dishwashing / Scouring Pads',
            'garbage bag': 'Grocery / Paper & Plastic',
            'cleaning gloves': 'Grocery / Cleaning Tools',
            'tobacco': 'Grocery / Tobacco-Related Products',
            'snack': 'Grocery',
            'rice': 'Grocery',
            'cooking ingredient': 'Grocery',
            'novel': 'Books, Movies and Music / Fiction / Adult Fiction',
            'religious book': 'Books, Movies and Music / Religion',
            'self help book': 'Books, Movies and Music / Motivational & Self-Help',
            'textbook': 'Books, Movies and Music / Education & Learning',
            "children's book": 'Books, Movies and Music / Fiction / Children & Teens',
            'dvd': 'Books, Movies and Music / DVDs',
            'notebook': 'Books, Movies and Music / Journals & Planners',
            'pen': 'Books, Movies and Music / Stationery / School Supplies',
            'pencil': 'Books, Movies and Music / Stationery / School Supplies',
            'crayon': 'Books, Movies and Music / Stationery / School Supplies',
            'marker': 'Books, Movies and Music / Stationery / School Supplies',
            'file folder': 'Home & Office / Office Products',
            'calculator': 'Home & Office / Office Products',
            'soldering iron': 'Industrial & Scientific / Industrial Electrical',
            'multimeter': 'Industrial & Scientific / Test, Measure & Inspect',
            'cable tie': 'Industrial & Scientific / Industrial Hardware',
            'safety vest': 'Industrial & Scientific / Occupational Health & Safety Products',
            'safety helmet': 'Industrial & Scientific / Occupational Health & Safety Products',
            'work gloves': 'Industrial & Scientific / Occupational Health & Safety Products',
            'safety boots': 'Industrial & Scientific / Occupational Health & Safety Products',
            'fire extinguisher': 'Industrial & Scientific / Occupational Health & Safety Products',
            'first aid': 'Health & Beauty / Medical Supplies & Equipment',
            'industrial tape': 'Industrial & Scientific / Tapes, Adhesives & Sealants',
            'spirit level': 'Industrial & Scientific / Industrial Power & Hand Tools',
            'measuring tape': 'Industrial & Scientific / Test, Measure & Inspect',
            'power drill': 'Home & Office / Tools & Home Improvement / Power Tools',
            'screwdriver': 'Home & Office / Tools & Home Improvement / Hand Tools',
            'wrench': 'Home & Office / Tools & Home Improvement / Hand Tools',
            'hammer': 'Home & Office / Tools & Home Improvement / Hand Tools',
            'saw': 'Home & Office / Tools & Home Improvement / Hand Tools',
            'pliers': 'Home & Office / Tools & Home Improvement / Hand Tools',
            'tool set': 'Home & Office / Tools & Home Improvement / Hand Tools',
            '3d printer': 'Industrial & Scientific / Additive Manufacturing Products / 3D Printers',
            'lab coat': 'Fashion / Uniforms, Work & Safety / Clothing / Medical',
            'lab equipment': 'Industrial & Scientific / Lab & Scientific Products',
            'smartphone': 'Phones & Tablets / Mobile Phones / Smartphones',
            'sim card': 'Phones & Tablets / Mobile Phones / Cell Phones / SIM Cards',
            'ipad': 'Phones & Tablets / Tablets',
            'screen protector': 'Phones & Tablets / Accessories / Screen Protectors',
            'phone holder': 'Phones & Tablets / Accessories',
            'desktop computer': 'Computing / Computers & Accessories',
            'printer': 'Computing / Computer Accessories',
            'scanner': 'Computing / Computer Accessories',
            'webcam': 'Computing / Computer Accessories / Audio & Video Accessories',
            'network switch': 'Computing / Computer Accessories / Networking Accessories',
            'ups battery backup': 'Computing / Computer Accessories',
            'ram': 'Computing / Computers & Accessories / Computer Components',
            'graphics card': 'Computing / Computers & Accessories / Computer Components',
            'cpu processor': 'Computing / Computers & Accessories / Computer Components / CPU Processors',
            'men jeans': "Fashion / Men's Fashion / Clothing / Jeans",
            'men trousers': "Fashion / Men's Fashion / Clothing / Pants",
            'men shorts': "Fashion / Men's Fashion / Clothing / Shorts",
            'men shirt': "Fashion / Men's Fashion / Clothing / Shirts",
            'men suit': "Fashion / Men's Fashion / Clothing / Suits",
            'men jacket': "Fashion / Men's Fashion / Clothing / Jackets & Coats",
            'men hoodie': "Fashion / Men's Fashion / Clothing / Fashion Hoodies & Sweatshirts",
            'men underwear': "Fashion / Men's Fashion / Clothing / Sleep & Lounge",
            'men socks': "Fashion / Men's Fashion / Accessories / Socks",
            'men cap': "Fashion / Men's Fashion / Accessories / Hats & Caps",
            'men belt': "Fashion / Men's Fashion / Accessories / Belts",
            'men tie': "Fashion / Men's Fashion / Accessories / Neckties",
            'men wallet': "Fashion / Men's Fashion / Accessories / Wallets, Card Cases & Money Organizers",
            'men watch': "Fashion / Watches & Sunglasses / Men's Watches",
            'men t-shirt': "Fashion / Men's Fashion / Clothing / Shirts",
            'men singlet': "Fashion / Men's Fashion / Clothing / Shirts",
            'men shoes': "Fashion / Men's Fashion / Shoes",
            'men sandals': "Fashion / Men's Fashion / Shoes / Sandals & Slides",
            'men boots': "Fashion / Men's Fashion / Shoes / Boots",
            'men fabric': "Fashion / Fabrics / Men's Fabric",
            'african men wear': "Fashion / Traditional & Cultural Wear / African",
            'men sleepwear': "Fashion / Men's Fashion / Clothing / Sleep & Lounge",
            'girls dress': "Fashion / Kids Fashion / Girls / Clothing",
            'girls skirt': "Fashion / Kids Fashion / Girls / Clothing",
            'girls top': "Fashion / Kids Fashion / Girls / Clothing",
            'boys shirt': "Fashion / Kids Fashion / Boys / Clothing",
            'boys trousers': "Fashion / Kids Fashion / Boys / Clothing",
            'boys shorts': "Fashion / Kids Fashion / Boys / Clothing",
            'school uniform': "Fashion / Kids Fashion / Boys / School Uniforms",
            'kids clothing': "Fashion / Kids Fashion / Boys / Clothing",
            'kids socks': "Fashion / Kids Fashion / Boys / Clothing",
            'kids cap': "Fashion / Kids Fashion / Boys / Accessories",
            'kids sleepwear': "Fashion / Kids Fashion / Girls / Clothing / Sleepwear & Robes",
            'ankara wear': "Fashion / Traditional & Cultural Wear / African",
            'traditional wear': "Fashion / Traditional & Cultural Wear / African",
            'aso-oke fabric': "Fashion / Fabrics / Women's Fabric / Aso oke",
            'television': "Electronics / Television & Video / Televisions",
            'sound bar': "Electronics / Home Audio / Home Theater Systems / Sound Bars",
            'home theater': "Electronics / Home Audio / Home Theater Systems",
            'speaker': "Electronics / Home Audio / Speakers / Bluetooth Speakers",
            'earphones': "Electronics / Portable Audio & Video / Headphones",
            'headset': "Electronics / Portable Audio & Video / Headphones",
            'camera': "Electronics / Cameras / Video Cameras",
            'action camera': "Electronics / Cameras / Video Cameras",
            'security camera': "Electronics / Cameras / Security & Surveillance Cameras",
            'projector': "Electronics / Television & Video / Video Projectors",
            'gps device': "Electronics / GPS & Navigation",
            'two-way radio': "Electronics / Radios & Transceivers",
            'radio': "Electronics / Home Audio / Compact Radios & Stereos",
            'dvd player': "Electronics / Television & Video / DVD Players",
            'streaming device': "Electronics / Television & Video",
            'vr headset': "Electronics / Wearable Technology",
            'drone': "Electronics / Camera & Photo / Drones",
            'sofa': "Home & Office / Home & Furniture / Sofas & Armchairs",
            'wardrobe': "Home & Office / Home & Furniture / Armoires & Wardrobes",
            'dressing table': "Home & Office / Home & Furniture / Vanities",
            'side table': "Home & Office / Home & Furniture / Coffee & End Tables",
            'dining table': "Home & Office / Home & Furniture / Kitchen & Dining Room Furniture / Kitchen & Dining Room Tables",
            'tv stand': "Home & Office / Home & Furniture / TV & Media Furniture",
            'mirror': "Home & Office / Home & Furniture / Mirrors",
            'wall art': "Home & Office / Home & Furniture / Wall Art",
            'clock': "Home & Office / Home & Kitchen / Home Décor / Clocks",
            'vase': "Home & Office / Home & Kitchen / Home Décor / Vases",
            'photo frame': "Home & Office / Home & Kitchen / Home Décor / Picture Frames",
            'curtain': "Home & Office / Home & Furniture / Window Treatments / Curtains & Drapes",
            'tablecloth': "Home & Office / Home & Kitchen / Kitchen & Dining / Table Linens",
            'diffuser': "Home & Office / Home & Kitchen / Heating, Cooling & Air Quality / Humidifiers",
            'lamp': "Home & Office / Tools & Home Improvement / Lighting & Ceiling Fans",
            'ceiling light': "Home & Office / Tools & Home Improvement / Lighting & Ceiling Fans",
            'led strip light': "Home & Office / Tools & Home Improvement / Lighting & Ceiling Fans",
            'solar panel': "Home & Office / Home & Kitchen / Home Energy",
            'inverter': "Home & Office / Home & Kitchen / Home Energy",
            'vacuum cleaner': "Home & Office / Appliances / Small Appliances / Vacuum Cleaners",
            'clothes iron': "Home & Office / Appliances / Small Appliances / Ironing & Laundry",
            'sewing machine': "Home & Office / Arts, Crafts & Sewing / Sewing",
            'knitting needles': "Home & Office / Arts, Crafts & Sewing / Knitting & Crochet",
            'stapler': "Home & Office / Office Products / Office Supplies & Equipment",
            'paper shredder': "Home & Office / Office Products / Office Supplies & Equipment",
            'laminator': "Home & Office / Office Products / Office Supplies & Equipment",
            'label printer': "Home & Office / Office Products / Office Supplies & Equipment",
            'hair extension': "Health & Beauty / Beauty & Personal Care / Hair Care / Extensions, Wigs & Accessories",
            'wig': "Health & Beauty / Beauty & Personal Care / Hair Care / Extensions, Wigs & Accessories",
            'hair attachment': "Health & Beauty / Beauty & Personal Care / Hair Care / Extensions, Wigs & Accessories",
            'hair dryer': "Health & Beauty / Beauty & Personal Care / Hair Care / Hair Cutting Tools",
            'hair straightener': "Health & Beauty / Beauty & Personal Care / Hair Care / Styling Tools & Appliances",
            'hair curler': "Health & Beauty / Beauty & Personal Care / Hair Care / Styling Tools & Appliances",
            'nail polish': "Health & Beauty / Beauty & Personal Care / Makeup / Nails",
            'nail kit': "Health & Beauty / Beauty & Personal Care / Makeup / Nails",
            'lipstick': "Health & Beauty / Beauty & Personal Care / Makeup / Lips",
            'foundation': "Health & Beauty / Beauty & Personal Care / Makeup / Face",
            'concealer': "Health & Beauty / Beauty & Personal Care / Makeup / Face",
            'highlighter makeup': "Health & Beauty / Beauty & Personal Care / Makeup / Face",
            'blush': "Health & Beauty / Beauty & Personal Care / Makeup / Cheeks",
            'eyeliner': "Health & Beauty / Beauty & Personal Care / Makeup / Eyes",
            'mascara': "Health & Beauty / Beauty & Personal Care / Makeup / Eyes",
            'eyeshadow': "Health & Beauty / Beauty & Personal Care / Makeup / Eyes",
            'makeup brush': "Health & Beauty / Beauty & Personal Care / Makeup / Makeup Brushes",
            'deodorant': "Health & Beauty / Beauty & Personal Care / Personal Care / Deodorants & Antiperspirants",
            'body spray': "Health & Beauty / Beauty & Personal Care / Fragrance",
            'hair clipper': "Health & Beauty / Beauty & Personal Care / Hair Care / Hair Cutting Tools",
            'electric razor': "Health & Beauty / Beauty & Personal Care / Shaving & Hair Removal",
            'blood pressure monitor': "Health & Beauty / Medical Supplies & Equipment / Diagnostic, Monitoring & Test Equipment",
            'glucometer': "Health & Beauty / Medical Supplies & Equipment / Diagnostic, Monitoring & Test Equipment",
            'pulse oximeter': "Health & Beauty / Medical Supplies & Equipment / Diagnostic, Monitoring & Test Equipment",
            'thermometer': "Health & Beauty / Medical Supplies & Equipment / Diagnostic, Monitoring & Test Equipment",
            'back support': "Health & Beauty / Medical Supplies & Equipment / Braces, Splints & Supports / Back, Neck & Shoulder Supports",
            'compression socks': "Health & Beauty / Medical Supplies & Equipment / Braces, Splints & Supports / Compression Socks",
            'wheelchair': "Health & Beauty / Medical Supplies & Equipment / Daily Living Aids",
            'walking aid': "Health & Beauty / Medical Supplies & Equipment / Daily Living Aids",
            'crutches': "Health & Beauty / Medical Supplies & Equipment / Daily Living Aids",
            'sanitary pad': "Health & Beauty / Beauty & Personal Care / Personal Care / Feminine Hygiene",
            'tampon': "Health & Beauty / Beauty & Personal Care / Personal Care / Feminine Hygiene",
            'feminine wash': "Health & Beauty / Beauty & Personal Care / Personal Care / Feminine Hygiene",
            'condom': "Health & Beauty / Sexual Wellness",
            'lubricant': "Health & Beauty / Sexual Wellness",
            'massage gun': "Health & Beauty / Wellness & Relaxation / Massage Equipment",
            'massage chair': "Health & Beauty / Wellness & Relaxation / Massage Equipment",
            'electric massager': "Health & Beauty / Wellness & Relaxation / Massage Equipment",
            'weight loss supplement': "Health & Beauty / Vitamins & Dietary Supplements / Supplements",
            'protein powder': "Health & Beauty / Sports Nutrition",
            'sports supplement': "Health & Beauty / Sports Nutrition",
            'fish oil': "Health & Beauty / Vitamins & Dietary Supplements / Supplements / Fish Oil",
            '

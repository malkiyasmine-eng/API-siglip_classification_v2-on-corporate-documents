# classifier.py
# ─────────────────────────────────────────────────────────────────────────────
# Inference module — refactored from notebook
# google/siglip-base-patch16-256-multilingual
# ─────────────────────────────────────────────────────────────────────────────

# ── Section 1: Imports ────────────────────────────────────────────────────────
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel

# ── Section 2: Device & Model ID ─────────────────────────────────────────────
MODEL_ID = "google/siglip-base-patch16-256-multilingual"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

# Global references — These are filled ONLY once in load_model().
processor     = None   #tokenizer + image preprocessor
model         = None   #SigLIP neural network
text_matrix   = None   # Pre-computed text embeddings (C, D), one row per class

# ── Section 3: CLASS_NAMES & PROMPTS) ──────────
#output labels (The model can ONLY predict these labels:)
CLASS_NAMES = [
    'NIF_certificate',
    'NIS_certificate',
    'certificat_existence',
    'tax_declaration_form',
    'residence_certificate',
    'legal_contract_front',
    'legal_contract_inside',
    'balance_sheet',
    'RC_front',
    'RC_inside_activities',
    'RC_inside_2',
    'driving_license_front',
    'driving_license_back',
    'driving_license_frontback',
]
#SigLIP learns: how well image matches each text
PROMPTS = {

    # ───────────────────────────── NIF CERTIFICATE ─────────────────────────────
    'NIF_certificate': [
        'an Algerian DGI tax identification certificate شهادة التسجيل الجبائي with mostly empty white space, '
        'the top left has a small building illustration and text DIRECTION GENERALE DES IMPOTS المديرية العامة للضرائب, '
        'two stacked rectangular bordered boxes in the upper center contain '
        'ATTESTATION D IMMATRICULATION FISCALE and NUMERO D IDENTIFICATION FISCALE NIF رقم التعريف الجبائي, '
        'below the boxes only a few sparse label-value lines: Raison Sociale or NOM Prenom, '
        'a small separate box contains only the NIF number digits, '
        'the lower half of the page is entirely blank empty white space, '
        'NO full page outer border frame, NO barcode strip, NO ONS logo, NO bilingual two-column layout, '
        'approximately seventy percent of total page area is empty white space',

        'DGI Algeria fiscal registration certificate with extreme white space, '
        'DIRECTION GENERALE DES IMPOTS المديرية العامة للضرائب at top left with building icon, '
        'two stacked boxes ATTESTATION D IMMATRICULATION FISCALE and NIF رقم التعريف الجبائي, '
        'only two or three label-value lines below the boxes, '
        'NIF number in a small separate box, '
        'no barcode no outer border no bilingual columns no dense Arabic paragraphs, '
        'most of the page is blank white',
    ],

    # ───────────────────────────── NIS CERTIFICATE ─────────────────────────────
    'NIS_certificate': [
        'an Algerian ONS statistical registration document عدد التعريف الإحصائي, '
        'THE ENTIRE PAGE IS ENCLOSED WITHIN A FULL-PAGE OUTER RECTANGULAR BORDER FRAME running along all four edges, '
        'a circular logo with triangle and letters ONS المديرية الوطنية للإحصاء appears in one top corner inside the border, '
        'a thick-bordered rectangle contains a bold Arabic title for the statistical identification notice, '
        'DIRECTLY BELOW THE TITLE BOX is a dense block of small Arabic legal text — '
        'this paragraph between the title and the NIS number is unique to this document type, '
        'the NIS number رقم التعريف الإحصائي appears as spaced digit groups, '
        'a bilingual two-column layout shows French labels on one side and Arabic labels on the other, '
        'French labels include NOM OU RAISON SOCIAL ADRESSE WILAYA COMMUNE CODE ACTIVITE, '
        'a circular ONS stamp and footer with ONS.DZ website address appear at the bottom, '
        'THIS DOCUMENT HAS A FULL PAGE OUTER BORDER FRAME which NIF certificates never have',

        'ONS Algeria statistical identification notice with full rectangular border frame enclosing all content, '
        'circular ONS logo top corner, thick title box with Arabic NIS title, '
        'dense Arabic legal paragraph block between title and NIS number رقم التعريف الإحصائي, '
        'bilingual French and Arabic two-column data layout, '
        'ONS stamp and ONS.DZ footer, full page border frame is the primary identifier',
    ],

    # ───────────────────────── CERTIFICAT D'EXISTENCE ──────────────────────────
    'certificat_existence': [
        'a scanned Algerian DGI existence certificate, '
        'the top right area contains the printed form code Serie C n 20 with an underline, '
        'the upper right quadrant has the large bold printed word CERTIFICAT, '
        'below CERTIFICAT is handwritten cursive text for the word Existence or similar, '
        'the left side has a stacked hierarchy of printed text: REPUBLIQUE ALGERIENNE MINISTERE DES FINANCES DIRECTION GENERALE DES IMPOTS and a local inspection name, '
        'the center of the page contains a row of individual empty square boxes labeled N I F for entering one digit per box, '
        'below the NIF boxes are a few handwritten lines on dotted leaders followed by blank dotted rows, '
        'the blank dotted rows are crossed out by one or more large diagonal pen strokes preventing additions, '
        'the diagonal slash lines are the most visually prominent feature in the lower half of the page, '
        'a circular inspection stamp and handwritten signature appear at the bottom, '
        'small vertical text is printed sideways along the left margin',

        'a photo of a Serie C administrative form where large diagonal pen strokes slash through blank dotted lines in the lower half, '
        'the word CERTIFICAT appears in large bold print in the upper right area, '
        'a row of small individual printed square boxes labeled N I F appears in the center, '
        'a few handwritten lines on dotted leaders appear above the diagonal slash zone, '
        'the upper left has a stacked DGI ministry hierarchy with a circular stamp overlapping it, '
        'a signature and circular stamp appear at the bottom of the page',
    ],

    # ───────────────────────── TAX DECLARATION FORM ────────────────────────────
    'tax_declaration_form': [
        'a photo of an Algerian tax administration form with the large bold printed title DECLARATION D EXISTENCE at the top, '
        'the upper section is a dense questionnaire with many rows of dotted leader lines filled with handwritten cursive answers, '
        'two long rows of small connected individual printed square boxes appear in the upper section, '
        'one row of digit boxes is labeled NIS and the other row is labeled NIF, '
        'each individual printed square box contains a single handwritten digit, '
        'an empty dotted Date de Reception box appears in the upper right corner, '
        'a printed selector paragraph mentions impot sur les benefices IBS and impot sur le revenu IRG',

        'a photo of a completed Algerian tax form Serie G 8 densely covered in handwritten text, '
        'the most prominent printed element is a large heavily bordered rectangle titled FORME JURIDIQUE DE L ENTREPRISE, '
        'inside this rectangle is a two-column checklist of legal entity types such as Entreprise individuelle SARL Societe par actions, '
        'one of the small printed square checkboxes in the checklist is marked with a handwritten X or checkmark, '
        'the areas above and below the checklist are filled with dense handwritten answers on dotted lines, '
        'a large circular Inspection des Impots stamp overlaps a signature in the bottom right corner',
    ],

    # ───────────────────────── RESIDENCE CERTIFICATE ───────────────────────────
    'residence_certificate': [
        'an Algerian municipal residence document with the large bold Arabic title بطاقة إقامة meaning residence card, '
        'the top right has a four-line Arabic administrative header listing الجمهورية الجزائرية وزارة الداخلية and wilaya daira commune, '
        'some versions have a horizontal barcode strip directly below the بطاقة إقامة title and some do not, '
        'the document body has sparse widely spaced Arabic text rows on dotted leader lines, '
        'the Arabic phrase نشهد بأن meaning we certify that appears followed by personal data, '
        'fields include السيد full name المولود date and place of birth السكن address, '
        'Arabic text states the person has lived يقيم بنفس العنوان منذ أكثر من ستة 6 أشهر for more than 6 months, '
        'a circular municipal stamp رئيس المجلس الشعبي البلدي and handwritten signature appear at the bottom left, '
        'a Latin-script transliteration of the persons full name is printed at the very bottom, '
        'the page is a full A4 document with large white space not a small card',

        'scanned Algerian bataqa iqama بطاقة إقامة residence certificate, '
        'bold Arabic title بطاقة إقامة in center upper area, '
        'sparse Arabic rows with dotted fill lines for personal data, '
        'Arabic certification phrase نشهد بأن in the body, '
        'circular commune stamp and signature at bottom left, '
        'Latin name printed at bottom, full A4 page with white space',
    ],

    # ───────────────────────── LEGAL CONTRACT FRONT ────────────────────────────
    'legal_contract_front': [
        'a notarial legal contract cover page عقد توثيقي in Arabic, '
        'THE PAGE HAS EITHER decorative starburst fan-shaped ornaments in the four corners '
        'OR a thick ornate decorative border frame running along all four edges of the page, '
        'the interior between the ornaments or border is mostly empty white space, '
        'sparse centered Arabic text includes رقم الفهرس index number and handwritten date, '
        'طبيعة العقد nature of contract appears on a dotted line, '
        'الأطراف parties section lists two names as السيد and والسيد, '
        'the bottom has a dark bordered box with the notary name and address مكتب التوثيق, '
        'NO barcode NO QR code NO dense paragraph text filling the page '
        'NO bilingual columns and NO printed barcode strip under any title',

        'Arabic notarial contract cover page with decorative corner ornaments or ornate full border, '
        'mostly blank white interior with sparse Arabic text رقم الفهرس طبيعة العقد الأطراف, '
        'notary office box مكتب التوثيق at bottom with address and phone, '
        'no barcodes no dense paragraphs no grid tables',
    ],

    # ───────────────────────── LEGAL CONTRACT INSIDE ───────────────────────────
    'legal_contract_inside': [
        'an interior page of an Arabic notarial legal contract, '
        'the page is densely filled with typed Arabic legal prose organized into numbered clauses or articles, '
        'Arabic legal terminology such as المادة article or البند clause begins each numbered paragraph, '
        'the text covers the page from top to bottom in continuous dense paragraphs, '
        'a circular notary stamp خاتم التوثيق and handwritten signature appear at the bottom, '
        'a rectangular registration tax stamp may appear in a corner, '
        'NO vertical ruled line dividing the page into margin and text columns, '
        'NO rotated serial number in a margin, '
        'NO asterisk strings, NO grid tables, NO QR code, NO barcode',

        'dense Arabic legal prose filling entire page in numbered clauses المادة البند, '
        'notary stamp خاتم التوثيق and signature at bottom, '
        'no vertical dividing line no margin column no serial number no asterisks no tables',
    ],

    # ───────────────────────── BALANCE SHEET ───────────────────────────────────
    'balance_sheet': [
        'a photo of a French-language accounting financial statement page, '
        'the page header contains company identification data including company name NIF number and fiscal year end date in French, '
        'a shaded or filled header row contains one of these accounting titles: BILAN PASSIF or BILAN ACTIF or COMPTE DE RESULTAT or TABLEAU DES COMPTES DE RESULTAT, '
        'below the title row is a structured table entirely in French, '
        'the left column contains French accounting category labels such as CAPITAUX PROPRES PASSIFS NON-COURANTS PASSIFS COURANTS or Ventes de marchandises, '
        'the right columns contain large monetary numeric values representing financial amounts, '
        'bold rows labeled TOTAL I or TOTAL II or TOTAL GENERAL PASSIF appear as subtotal separators, '
        'the entire main table body is in French with no Arabic text in the financial data rows',

        'a scanned French corporate financial filing page, '
        'the very top left contains a small bordered box with the text IMPRIME DESTINE A L ADMINISTRATION, '
        'a NIF identification number appears near the top right in a two-cell table, '
        'the company name activity and address are listed in a header block below the NIF, '
        'the main body is a multi-column French accounting table with column headers N and N-1 representing current and prior year, '
        'French accounting terms like Capital emis Reserves Emprunts Fournisseurs Impots Tresorerie appear as row labels, '
        'the page is entirely in French and contains no Arabic text in the table body',
    ],

    # ───────────────────────── RC FRONT ────────────────────────────────────────
    'RC_front': [
        'an Algerian CNRC السجل التجاري commercial register cover page in PORTRAIT VERTICAL ORIENTATION taller than wide, '
        'light green tinted background with subtle security dot pattern, '
        'the CNRC logo and text الجمهورية الجزائرية السجل التجاري الوطني appear in the top right corner, '
        'a large rectangle with very rounded pill-shaped corners dominates the center of the page, '
        'inside the rounded rectangle handwritten Arabic text shows the company name and legal form such as SARL or EURL, '
        'on the LEFT side a narrow vertical bordered panel contains dotted lines with registration number and dates in YYYY/MM/DD format, '
        'a square QR code labeled CNRC appears in the BOTTOM RIGHT corner, '
        'the page is mostly empty light green or white space with these three zones, '
        'this is portrait orientation taller than wide unlike landscape documents, '
        'it has a QR code not a linear barcode strip',

        'CNRC السجل التجاري commercial register first page portrait format taller than wide, '
        'light green background security pattern, '
        'large rounded rectangle in center with handwritten Arabic company name, '
        'narrow left panel with registration number and YYYY/MM/DD dates, '
        'QR code bottom right corner labeled CNRC, '
        'portrait orientation taller than wide',
    ],

    # ───────────────────────── RC INSIDE ACTIVITIES ────────────────────────────
    'RC_inside_activities': [
        'a photo of an Algerian commercial register data page divided into two distinct horizontal sections, '
        'the upper section is a multi-column table for directors and shareholders personal details, '
        'columns in the upper table contain names birth dates addresses nationalities and legal roles, '
        'the lower section is a bordered table with three columns for activity data, '
        'the leftmost narrow column shows 6-digit numeric activity codes such as 108113 or 612206, '
        'THE MOST DISTINCTIVE FEATURE IS THE WIDE MIDDLE COLUMN which contains Arabic text descriptions '
        'followed by LONG REPEATED STRINGS OF ASTERISK CHARACTERS *** that fill the remaining space after each description, '
        'the asterisk strings *** are highly conspicuous and visible across many rows of the lower table, '
        'the rightmost column may show a sector label, '
        'this page has a plain white background and no vertical colored line dividing the page, '
        'THIS IS NOT A BALANCE SHEET: it has Arabic text and asterisk strings, not French accounting terms',

        'a scanned CNRC register interior page on plain white paper, '
        'the page is split into an upper personal data grid and a lower activity code table, '
        'in the lower table each row shows a 6-digit number on the left and Arabic description text ending in multiple asterisks *** on the right, '
        'THE ASTERISK PATTERN *** IS THE SINGLE MOST RECOGNIZABLE FEATURE — long repetitive strings of asterisks fill the right portion of each row, '
        'the upper portion has a table with bordered cells containing Arabic names dates and nationalities, '
        'there is no vertical dividing line running down the page and no serial number printed in the margin',
    ],

    # ───────────────────────── RC INSIDE 2 ─────────────────────────────────────
    'RC_inside_2': [
        'an Algerian CNRC commercial register legal page with light green background tint, '
        'a vertical ruled line divides the page into a narrow LEFT margin and a wide RIGHT text area, '
        'a long 9 or 10 digit serial number such as 700164332 or 700170604 is printed ROTATED 90 DEGREES '
        'running vertically upward along the left margin — this rotated number is the most distinctive feature, '
        'a circular official stamp السجل التجاري الوطني with crescent and star emblem appears in the left margin, '
        'a handwritten signature appears in the left margin above or near the stamp, '
        'additional rectangular red or black ink stamps may appear in the left margin, '
        'the wide right column contains dense continuous Arabic legal prose paragraphs, '
        'a narrow header table with bordered cells spans the full width at the top of the page, '
        'the combination of green background plus rotated serial number plus circular stamp in left margin identifies this document',

        'CNRC register page green background تint, '
        'vertical ruled line separating narrow left margin from wide right text column, '
        'serial number 700XXXXXX printed sideways rotated 90 degrees in left margin, '
        'circular stamp السجل التجاري and signature in left margin, '
        'dense Arabic legal paragraphs filling the right column, '
        'header table at top spanning full width',
    ],

    # ───────────────────────── DRIVING LICENSE FRONTBACK ───────────────────────
    'driving_license_frontback': [
        'a composite image showing two Algerian driving license cards stacked vertically, '
        'the TOP HALF shows the FRONT card with a red and pink decorative guilloche security background pattern, '
        'bold letters DZ and the text DRIVING LICENSE and Arabic رخصة السيادة appear at the top of the front card, '
        'a portrait facial photograph of a person is on the left side of the front card, '
        'personal data fields with Latin names surname given name and date of birth appear on the right of the front card, '
        'the BOTTOM HALF shows the BACK card with a teal green or gray security background pattern, '
        'a square golden metallic smart chip is on the left side of the back card, '
        'a vehicle category grid table with rows labeled A A1 B C D BE CE is in the center of the back card, '
        'a small ghost portrait photo appears in the upper right of the back card, '
        'THREE LINES OF MACHINE READABLE ZONE text starting with DLDZAA followed by many less-than symbols <<< appear at the very bottom, '
        'the two cards are visually separated by a gap or border line in the middle of the image, '
        'the overall composite is TALLER THAN WIDE showing two complete cards',

        'a tall portrait image containing two driving license cards one above the other, '
        'upper card has red pink guilloche pattern with DZ DRIVING LICENSE رخصة السيادة text and a face photo, '
        'lower card has teal green background with metallic chip golden square on left and vehicle categories grid A B C D, '
        'bottom of image has machine readable zone DLDZAA followed by <<< chevron symbols across three lines, '
        'both cards visible simultaneously in one image with a dividing gap between them',
    ],

    # ───────────────────────── DRIVING LICENSE FRONT ───────────────────────────
    'driving_license_front': [
        'a single Algerian driving license card showing only the FRONT face, '
        'the card has a red and pink decorative guilloche security pattern background, '
        'bold letters DZ appear in the upper left corner of the card, '
        'the text DRIVING LICENSE and Arabic رخصة السيادة appear at the top, '
        'a color portrait photograph of a persons face occupies the left portion of the card, '
        'a small printed signature appears below the portrait photograph, '
        'the right side contains numbered data fields: surname family name given name date of birth and license categories, '
        'an 18-digit national identification number appears in field 4d, '
        'this image shows ONLY ONE CARD — there is no second card below it, '
        'there is NO metallic chip NO MRZ machine readable zone and NO <<< chevron text visible, '
        'the card is wider than tall in a standard credit card landscape format',

        'one single Algerian driving license front card with red guilloche security background, '
        'DZ and DRIVING LICENSE رخصة السيادة at top, '
        'face portrait photo on left half, numbered Latin text fields on right half, '
        'only one card is visible — no chip no category grid no <<< symbols anywhere in the image',
    ],

    # ───────────────────────── DRIVING LICENSE BACK ────────────────────────────
    'driving_license_back': [
        'a single Algerian driving license card showing only the BACK face, '
        'the card has a teal green or gray security pattern background, '
        'a square golden metallic smart chip appears on the LEFT side of the card, '
        'a vertical serial number is printed along the left edge next to the chip, '
        'a vehicle license category grid table fills the CENTER of the card with rows A A1 B C C1 D BE CE DE, '
        'each category row has date columns showing issue and expiry dates, '
        'a small faded ghost portrait photograph appears in the UPPER RIGHT corner, '
        'THREE LINES of machine readable zone text appear at the very bottom of the card, '
        'the first MRZ line starts with DLDZAA followed by digits and many <<< less-than symbols, '
        'this image shows ONLY ONE CARD — there is no front card stacked above it, '
        'there is NO portrait photo on the left and NO red guilloche background',

        'single driving license back card with teal green background, '
        'golden metallic square chip on left, category grid A B C D rows center, '
        'DLDZAA<<< machine readable text at bottom across three lines, '
        'small ghost photo top right, only one card visible no red background no face photo on left',
    ],
}


# ── Section 4: get_text_features() — copied verbatim from notebook ────────────
@torch.no_grad()
#converting raw text → numbers.
def get_text_features(prompts):
    inputs = processor(
        text=prompts, 
        return_tensors="pt", #PyTorch tensors
        padding="max_length", #All sentences must have SAME length
        truncation=True,
    ).to(DEVICE)
    out = model.get_text_features(**inputs) #SigLIP turns tokenized text into vectors, Each prompt becomes a vector like: [0.12, -0.98, 0.33, ..., 0.05]
    feats = out if isinstance(out, torch.Tensor) else out.pooler_output
    return feats / feats.norm(dim=-1, keepdim=True)
    

# ── Section 5: Model + text embeddings loader (called once at API startup) ────
def load_model():
    """
    Load the SigLIP processor + model, then pre-compute and cache
    all text embeddings. Call this once when the API starts.
    """
    global processor, model, text_matrix

    print(f"⚡ Device: {DEVICE}")
    print(f"🔄 Loading model: {MODEL_ID} …")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model = model.to(DEVICE).eval()

    print("✅ Model loaded — pre-computing text embeddings …")

    # Build averaged text feature matrix (C, D) — one row per class
    class_features = [] #This list will store: one vector per class
    for cls in CLASS_NAMES:
        feats = get_text_features(PROMPTS[cls]) # (n_prompts, D) n_prompts = number of descriptions, D = embedding size
        #GET TEXT FEATURES FOR EACH CLASS 
        #each sentence → embedding vector
        # result → multiple vectors per class
        class_features.append(feats.mean(dim=0, keepdim=True)) 
        # collapsing multiple descriptions into ONE vector (ONE final representation per class)
        #3 prompt vectors:[0.1, 0.2, 0.3]   after mean : [0.15, 0.18, 0.35]
                         # [0.2, 0.1, 0.4]
                        #[0.15, 0.25, 0.35]
    text_matrix = torch.cat(class_features, dim=0)        # (C, D)
    #stack all class vectors into a single matrix:
    text_matrix = text_matrix / text_matrix.norm(dim=-1, keepdim=True)

    print(f"✅ Text embeddings ready — {len(CLASS_NAMES)} classes")


# ── Section 6: Single-image inference function ────────────────────────────────
@torch.no_grad()
def classify_single_image(pil_image: Image.Image, top_k: int = 3, scale: float = 100.0) -> dict:
    """
    Classify a single PIL image using pre-computed text embeddings.

    Args:
        pil_image : PIL.Image in RGB mode
        top_k     : number of top predictions to return (default 3)
        scale     : logit scaling factor (same as notebook, default 100)

    Returns:
        {
            "label":      "<best class name>",
            "confidence": 0.87,          # float 0-1
            "top3": [
                {"label": "...", "confidence": 0.87},
                {"label": "...", "confidence": 0.08},
                {"label": "...", "confidence": 0.03},
            ]
        }
    """
    if model is None or processor is None or text_matrix is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    # Encode image (You convert image → tensor format)
    img_inputs = processor(images=pil_image, return_tensors="pt").to(DEVICE)
     
    #SigLIP converts image → embedding vector
    out = model.get_image_features(**img_inputs)

    img_feat = out if isinstance(out, torch.Tensor) else out.pooler_output
    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)       # (1, D)

    # Cosine similarity → softmax probabilities
    logits = (img_feat @ text_matrix.T) * scale     #dot product between: image vector & all class vectors  #example logits = [2.0, 8.0, 3.0]              # (1, C)
    probs  = logits.softmax(dim=-1).cpu().numpy()[0]   # (C,) #example probs = [0.0024, 0.9909, 0.0067]
    #Convert raw scores → probabilities

    # Top-k results
    top_indices = np.argsort(probs)[::-1][:top_k]
    top_results = [
        {"label": CLASS_NAMES[i], "confidence": float(round(probs[i], 6))}
        for i in top_indices
    ]

    return {
        "label":      top_results[0]["label"],
        "confidence": top_results[0]["confidence"],
        "top3":       top_results,
    }

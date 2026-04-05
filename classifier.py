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
    "NIF_certificate",
    "NIS_certificate",
    "certificat_existence",
    "tax_declaration_form",
    "residence_certificate",
    "legal_contract",
    "balance_sheet",
    "RC_front",
    "RC_inside_activities",
    "RC_inside_2",
    "driving_license_front",
    "driving_license_back",
    "driving_license_frontback",
]
#SigLIP learns: how well image matches each text
PROMPTS = {

    # ── NIF CERTIFICATE ───────────────────────────────────────────────────────
    "NIF_certificate": [
        # Anchor 1: Corporate version — DGI logo + two stacked boxes + extreme sparsity
        "a scanned Algerian DGI tax identification certificate with extreme white space, "
        "the top left corner contains a small square illustration of a government building, "
        "below this is the bold text DIRECTION GENERALE DES IMPOTS, "
        "the upper half of the page contains exactly two stacked rectangular bordered boxes centered on the page, "
        "the first box contains ATTESTATION D IMMATRICULATION FISCALE, "
        "the second smaller box contains NUMERO D IDENTIFICATION FISCALE NIF and loi de finances 2006, "
        "below the two boxes are only two label-value lines: Raison Sociale and Sigle, "
        "a third separate small box lower on the page contains only the NIF number starting with digits, "
        "the lower half of the page is entirely blank empty white space, "
        "there are no dotted fill-in lines no checkboxes and no questionnaire grid anywhere on the page",

        # Anchor 2: Individual version — NOM Prenom + bottom two-cell table
        "a scanned Algerian fiscal registration certificate almost entirely blank white, "
        "the top has two stacked rectangular boxes for ATTESTATION D IMMATRICULATION FISCALE and NUMERO D IDENTIFICATION FISCALE, "
        "the middle section has only three sparse label-value lines: NOM then Prenom then Date et lieu de naissance, "
        "the bottom of the page has a horizontal two-cell bordered table, "
        "the left cell is labeled Numero d Identification Fiscale NIF and the right cell contains a long number, "
        "a single signature and small circular stamp appear at the very bottom, "
        "the page has no dense text blocks no grid tables and no form questionnaire structure, "
        "approximately seventy percent of the total page area is empty white space",
    ],

    # ── NIS CERTIFICATE ───────────────────────────────────────────────────────
    "NIS_certificate": [
        # Anchor 1: Full-page outer border + ONS logo + thick title box + legal paragraph block
        "a scanned Algerian ONS statistical identification document, "
        "the entire page content is enclosed inside a full-page outer rectangular border frame, "
        "a circular logo containing a triangle and the letters O N S appears in the top right corner, "
        "a thick-bordered single rectangle in the upper center contains the bold Arabic title for identification notice, "
        "directly below the thick title box is a dense block of three or four lines of small Arabic legal text, "
        "this dense legal paragraph is a unique feature that appears between the title and the NIS number, "
        "the NIS number appears as a sequence of spaced digit groups such as 0 999 1615 11786 11, "
        "the data section shows two mirrored columns with French labels on one side and Arabic labels on the other, "
        "French labels include NOM OU RAISON SOCIAL SIGLE ADRESSE WILAYA COMMUNE CODE ACTIVITE NAA, "
        "a circular ONS stamp and a rectangular date stamp appear at the bottom, "
        "the page footer contains the website address ONS DZ and an email address",

        # Anchor 2: Outer frame + arrow symbol in NIS number box + footer URL
        "a photo of an Algerian statistical registration notice enclosed in a full outer rectangular page border, "
        "the ONS circular logo with triangle is in one of the top corners, "
        "a single thick-bordered rectangle contains the main Arabic title, "
        "a dense paragraph of small Arabic legal text appears immediately below the title box, "
        "the NIS identification number is displayed with an arrow symbol pointing to spaced digit groups, "
        "the bilingual data block lists company name address wilaya commune and activity code in two languages, "
        "the bottom footer contains text with the domain ONS DZ",
    ],

    # ── CERTIFICAT D'EXISTENCE ────────────────────────────────────────────────
    "certificat_existence": [
        # Anchor 1: Serie C n20 + CERTIFICAT heading + NIF square boxes + diagonal slashes
        "a scanned Algerian DGI existence certificate, "
        "the top right area contains the printed form code Serie C n 20 with an underline, "
        "the upper right quadrant has the large bold printed word CERTIFICAT, "
        "below CERTIFICAT is handwritten cursive text for the word Existence or similar, "
        "the left side has a stacked hierarchy of printed text: REPUBLIQUE ALGERIENNE MINISTERE DES FINANCES DIRECTION GENERALE DES IMPOTS and a local inspection name, "
        "the center of the page contains a row of individual empty square boxes labeled N I F for entering one digit per box, "
        "below the NIF boxes are a few handwritten lines on dotted leaders followed by blank dotted rows, "
        "the blank dotted rows are crossed out by one or more large diagonal pen strokes preventing additions, "
        "the diagonal slash lines are the most visually prominent feature in the lower half of the page, "
        "a circular inspection stamp and handwritten signature appear at the bottom, "
        "small vertical text is printed sideways along the left margin",

        # Anchor 2: Diagonal slashes as primary + NIF boxes + sparse layout
        "a photo of a Série C administrative form where large diagonal pen strokes slash through blank dotted lines in the lower half, "
        "the word CERTIFICAT appears in large bold print in the upper right area, "
        "a row of small individual printed square boxes labeled N I F appears in the center, "
        "a few handwritten lines on dotted leaders appear above the diagonal slash zone, "
        "the upper left has a stacked DGI ministry hierarchy with a circular stamp overlapping it, "
        "a signature and circular stamp appear at the bottom of the page",
    ],

    # ── TAX DECLARATION FORM ──────────────────────────────────────────────────
    "tax_declaration_form": [
        # Anchor 1: DECLARATION D'EXISTENCE title + NIS/NIF digit box rows + dense fill
        "a photo of an Algerian tax administration form with the large bold printed title DECLARATION D EXISTENCE at the top, "
        "the upper section is a dense questionnaire with many rows of dotted leader lines filled with handwritten cursive answers, "
        "two long rows of small connected individual printed square boxes appear in the upper section, "
        "one row of digit boxes is labeled NIS and the other row is labeled NIF, "
        "each individual printed square box contains a single handwritten digit, "
        "an empty dotted Date de Reception box appears in the upper right corner, "
        "a printed selector paragraph mentions impot sur les benefices IBS and impot sur le revenu IRG",

        # Anchor 2: FORME JURIDIQUE checklist block + dense handwriting + Inspection stamp
        "a photo of a completed Algerian tax form Serie G 8 densely covered in handwritten text, "
        "the most prominent printed element is a large heavily bordered rectangle titled FORME JURIDIQUE DE L ENTREPRISE, "
        "inside this rectangle is a two-column checklist of legal entity types such as Entreprise individuelle SARL Societe par actions, "
        "one of the small printed square checkboxes in the checklist is marked with a handwritten X or checkmark, "
        "the areas above and below the checklist are filled with dense handwritten answers on dotted lines, "
        "a large circular Inspection des Impots stamp overlaps a signature in the bottom right corner",
    ],

    # ── RESIDENCE CERTIFICATE ─────────────────────────────────────────────────
    "residence_certificate": [
        # Anchor 1: Arabic title box + barcode strip immediately below + 4-line Arabic header
        "a photo of an Algerian municipal residence document, "
        "the top right area has a stacked Arabic header of exactly four lines listing ministry wilaya daira and commune, "
        "the upper center contains a prominent rounded rectangular box with bold Arabic title text for residence card, "
        "immediately below this title box and directly touching it is a horizontal printed barcode strip, "
        "the barcode and the title box together form a single visual unit in the upper center of the page, "
        "the document body has widely spaced rows ending in dotted leaders with typed personal data, "
        "data fields include full name birth date birth place and residential address in Arabic, "
        "a dark circular municipal commune stamp overlaps a signature in the lower section, "
        "the page layout is sparse with significant white space between data rows",

        # Anchor 2: Barcode as unmistakable anchor + sparse Arabic layout
        "a scanned Algerian municipal document where a printed barcode strip appears directly beneath a rounded rectangle title box, "
        "the barcode is the single most distinctive printed element and is positioned immediately under the Arabic title, "
        "the top right corner has a four-line Arabic administrative hierarchy header, "
        "the body contains spaced-out Arabic label-value rows with dotted fill lines for personal identification data, "
        "a circular commune seal stamp is visible in the lower left area of the page",
    ],

    # ── LEGAL CONTRACT (Statuts) ──────────────────────────────────────────────
    "legal_contract": [
        # Anchor 1: Cover page — four corner starburst ornaments
        "a photo of a notarial legal contract cover page, "
        "the most visually distinctive feature is four identical decorative starburst or fan-shaped graphic ornaments in the four corners of the page, "
        "these corner ornaments are radiating geometric designs that look like sunbursts or folded fans, "
        "the interior of the page between the corner ornaments is mostly empty white space, "
        "sparse content includes a handwritten reference number labeled raqm al-fihris and a handwritten date, "
        "a line labeled tabiat al-aqd shows the nature of the contract in handwritten text on a dotted line, "
        "a section labeled al-atraf lists two parties as al-sayyid and wal-sayyid on dotted lines, "
        "the bottom quarter of the page has a dark bordered box with bold Arabic text showing a notary name and office address, "
        "a telephone number appears at the very bottom of the notary box",

        # Anchor 2: Interior page — numbered Arabic legal clauses + registration stamp
        "a photo of an interior page of a notarial Arabic legal document, "
        "the page is densely filled with typed Arabic paragraphs organized as numbered clauses, "
        "each numbered clause begins with an Arabic or Western numeral in parentheses followed by a paragraph of legal text, "
        "the page contains no tables no checkboxes no dotted fill-in lines and no bilingual columns, "
        "it consists entirely of numbered prose paragraphs in Arabic script, "
        "a rectangular ink registration or taxation stamp may appear in the top right margin, "
        "a circular notary seal overlaps a handwritten signature at the bottom of the page, "
        "a handwritten index number such as raqm al-fihris appears in the top right corner",
    ],

    # ── BALANCE SHEET (Liasse Fiscale) ────────────────────────────────────────
    "balance_sheet": [
        # Anchor 1: French financial table with BILAN or COMPTE DE RESULTAT header row
        "a photo of a French-language accounting financial statement page, "
        "the page header contains company identification data including company name NIF number and fiscal year end date in French, "
        "a shaded or filled header row contains one of these accounting titles: BILAN PASSIF or BILAN ACTIF or COMPTE DE RESULTAT or TABLEAU DES COMPTES DE RESULTAT, "
        "below the title row is a structured table entirely in French, "
        "the left column contains French accounting category labels such as CAPITAUX PROPRES PASSIFS NON-COURANTS PASSIFS COURANTS or Ventes de marchandises, "
        "the right columns contain large monetary numeric values representing financial amounts, "
        "bold rows labeled TOTAL I or TOTAL II or TOTAL GENERAL PASSIF appear as subtotal separators, "
        "the entire main table body is in French with no Arabic text in the financial data rows",

        # Anchor 2: IMPRIME DESTINE A L'ADMINISTRATION header + NIF box + French columns
        "a scanned French corporate financial filing page, "
        "the very top left contains a small bordered box with the text IMPRIME DESTINE A L ADMINISTRATION, "
        "a NIF identification number appears near the top right in a two-cell table, "
        "the company name activity and address are listed in a header block below the NIF, "
        "the main body is a multi-column French accounting table with column headers N and N-1 representing current and prior year, "
        "French accounting terms like Capital emis Reserves Emprunts Fournisseurs Impots Tresorerie appear as row labels, "
        "the page is entirely in French and contains no Arabic text in the table body",
    ],

    # ── RC FRONT (Cover Page) ─────────────────────────────────────────────────
    "RC_front": [
        # Anchor 1: THREE-PANEL LANDSCAPE + QR code + large rounded rectangle center
        "a photo of an Algerian CNRC commercial register cover page in landscape horizontal orientation, "
        "the document is wider than tall and divided into three vertical zones or panels side by side, "
        "one outer zone contains a square QR code block and a square CNRC building logo stacked vertically, "
        "the QR code block has the label CNRC printed next to or below it, "
        "the center zone contains a single large rectangle with very rounded corners containing a few lines of bold large Arabic text, "
        "the other outer zone contains a tall narrow bordered rectangle with dotted lines showing a registration number and two dates in YYYY MM DD format, "
        "the document has no handwritten fill-in fields no stamps and no dense paragraphs, "
        "the overall layout is minimalist with large areas of empty white space",

        # Anchor 2: QR + rounded center box + vertical date strip
        "a scanned CNRC commercial register document with a horizontal landscape orientation wider than tall, "
        "the three-panel side-by-side layout is the defining structural feature, "
        "a square QR code and a square CNRC logo appear together in one of the outer panels, "
        "the center panel has a large single rounded rectangle containing bold Arabic text for the register title, "
        "the third panel has a narrow bordered box with two YYYY MM DD format dates and a long registration number string, "
        "the page is plain white with no text-filled paragraphs and no handwriting anywhere",
    ],

    # ── RC INSIDE ACTIVITIES (Tabular Data Page) ──────────────────────────────
    "RC_inside_activities": [
        # Anchor 1: Directors grid TOP + asterisk activity table BOTTOM
        "a photo of an Algerian commercial register data page divided into two distinct horizontal sections, "
        "the upper section is a multi-column table for directors and shareholders personal details, "
        "columns in the upper table contain names birth dates addresses nationalities and legal roles, "
        "the lower section is a bordered table with three columns for activity data, "
        "the leftmost narrow column shows 6-digit numeric activity codes such as 108113 or 612206, "
        "the wide middle column contains Arabic text descriptions followed by long repeated strings of asterisk characters, "
        "the asterisk strings *** fill the remaining space after each Arabic activity description, "
        "the rightmost column may show a sector label, "
        "this page has a plain white background and no vertical colored line dividing the page",

        # Anchor 2: Asterisks + 6-digit codes as unmistakable signature
        "a scanned CNRC register interior page on plain white paper, "
        "the page is split into an upper personal data grid and a lower activity code table, "
        "in the lower table each row shows a 6-digit number on the left and Arabic description text ending in multiple asterisks *** on the right, "
        "the asterisk pattern *** is highly repetitive and visible across many rows, "
        "the upper portion has a table with bordered cells containing Arabic names dates and nationalities, "
        "there is no vertical dividing line running down the page and no serial number printed in the margin",
    ],

    # ── RC INSIDE 2 (Penalties/Legal Page) ────────────────────────────────────
    "RC_inside_2": [
        # Anchor 1: Vertical dividing line + LEFT narrow column with stamp + serial number
        "a photo of an Algerian commercial register legal information page, "
        "the most distinctive structural feature is a vertical line dividing the page into a narrow left column and a wide right column, "
        "the narrow left column contains a large circular official stamp with Arabic text and a crescent star emblem, "
        "below the circular stamp in the narrow left column is a long 9 or 10 digit serial number printed in isolation, "
        "the serial number such as 700170604 appears alone on its own line in the left margin area, "
        "a handwritten signature appears in the upper portion of the left column, "
        "the wide right column is filled with dense continuous Arabic prose paragraphs, "
        "a narrow header table spans the full width at the very top of the page, "
        "there are no asterisk strings no 6-digit activity codes and no directors personal data table on this page",

        # Anchor 2: Serial number + stamp in margin + dense prose (no asterisks)
        "a scanned CNRC commercial register page with a vertical ruled line separating a narrow margin column from a wide text column, "
        "a 9 or 10 digit printed serial number appears in the narrow left or right margin column, "
        "a large circular official stamp is visible in the margin column, "
        "the main wide column is entirely filled with dense typed Arabic paragraphs of continuous legal text, "
        "the text in the wide column has no numbered clause markers and no asterisk strings, "
        "a narrow bordered table appears at the top spanning the full page width",
    ],

    # ── DRIVING LICENSE FRONT ─────────────────────────────────────────────────
    "driving_license_front": [
        # Anchor 1: Single card + portrait photo LEFT + DZ + DRIVING LICENSE text
        "a photo of a single Algerian driving license card, "
        "the card has the large bold letters DZ in the upper left corner, "
        "the top edge of the card shows Arabic text and the English words DRIVING LICENSE, "
        "the left half of the card is dominated by a large color or grayscale portrait photograph of a persons face, "
        "a small digitized or printed signature appears directly below the portrait photograph, "
        "the right half contains numbered data fields with personal information including surname given name date of birth and license categories, "
        "one of the data fields shows an 18-digit identification number, "
        "the card has a decorative security background pattern with fine repeated lines, "
        "this is a single card and there is no second card stacked below it in the image",

        # Anchor 2: Portrait photo + numbered rows + single card geometry
        "a close up of a single horizontal ID-sized Algerian drivers license card, "
        "a large facial photograph occupies the left portion of the card, "
        "a small signature is printed below the portrait, "
        "the right portion lists numbered data rows including fields labeled with numbers 1 2 3 4a 4b 9, "
        "the card header shows DZ and DRIVING LICENSE in English, "
        "this image shows exactly one card with no other card stacked above or below it, "
        "the card has a fine security guilloche background pattern",
    ],

    # ── DRIVING LICENSE BACK ──────────────────────────────────────────────────
    "driving_license_back": [
        # Anchor 1: Chip + vehicle category grid + MRZ starting DLDZA
        "a photo of a single Algerian driving license back card, "
        "the left side of the card features a square metallic gold smart card chip, "
        "a vertical serial number is printed along the left margin next to the chip, "
        "the center of the card contains a rectangular grid table listing vehicle license categories A A1 B C D BE CE DE with date columns, "
        "each category row in the grid has corresponding date values for issue and expiry, "
        "a small secondary faded portrait photograph appears in the upper right corner of the card, "
        "the bottom third of the card contains a machine readable zone with three lines of dense text, "
        "the first line of the machine readable zone starts with the letters DLDZA followed by numbers and many chevron characters, "
        "this is a single card with no other card stacked above or below it in the image",

        # Anchor 2: MRZ + chip + category grid as triple anchor
        "a close up of a single horizontal Algerian driving license card reverse side, "
        "a gold or silver metallic chip is visible on one side, "
        "a vehicle category matrix table fills the center with rows for categories A B C D and date columns, "
        "three rows of machine readable zone text appear at the bottom, "
        "the first MRZ line begins with DLDZA and contains many chevron symbols, "
        "a small ghost secondary photo appears in a corner, "
        "this image contains exactly one card with no additional card above or below",
    ],

    # ── DRIVING LICENSE FRONT+BACK ────────────────────────────────────────────
    "driving_license_frontback": [
        # Anchor 1: Two-card stacked geometry + MRZ visible at bottom
        "a tall portrait-oriented composite image showing two Algerian driving license cards stacked vertically one above the other, "
        "the upper card shows a facial portrait photograph on one side and personal data fields including DZ and DRIVING LICENSE text, "
        "the lower card shows a square metallic chip on one side and a vehicle category grid table in the center, "
        "the very bottom of the composite image contains three lines of machine readable zone text starting with DLDZA and many chevron symbols, "
        "the two cards together make the overall image taller than wide with an aspect ratio of approximately one to two, "
        "both cards are visible simultaneously in the same image frame",

        # Anchor 2: Two-card geometry + chip on bottom card + MRZ at very bottom
        "a single tall image containing two ID cards one above the other, "
        "the top card has a portrait facial photo and numbered personal data fields, "
        "the bottom card has a gold metallic chip and a grid table of vehicle license categories, "
        "three rows of machine readable zone text with chevron characters appear at the very bottom of the image, "
        "the composite image is significantly taller than wide because it contains two complete cards stacked vertically, "
        "a small secondary thumbnail portrait photo is visible on the bottom card",
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

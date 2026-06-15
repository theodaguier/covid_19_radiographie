"""Genere les pages des nouvelles sections (Transfer Learning : methodologie justifiee,
ResNet50, EfficientNetB0, comparatif, interpretabilite) et les insere dans
'RAPPORT FINAL VF.pdf' apres la matrice de confusion du Boosting (fin de 8.6), avant
les conclusions. Tous les chiffres et parametres proviennent du notebook
Rattrapage (1).ipynb. Modele retenu : ResNet50 fine-tune (macro F1 0,936).
"""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether,
)
from pypdf import PdfReader, PdfWriter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "reports", "figures")
ORIG = os.path.join(ROOT, "RAPPORT FINAL VF.pdf")
NEWPAGES = os.path.join(ROOT, "reports", "_tl_sections_tmp.pdf")
OUT = ORIG

BLUE = colors.HexColor("#1F6FB2")
GREY = colors.HexColor("#444444")

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontName="Helvetica-Bold",
                    fontSize=14, textColor=BLUE, spaceBefore=16, spaceAfter=8)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName="Helvetica-Bold",
                    fontSize=11.5, textColor=BLUE, spaceBefore=10, spaceAfter=5)
H3 = ParagraphStyle("H3", parent=styles["Heading3"], fontName="Times-Bold",
                    fontSize=11, textColor=GREY, spaceBefore=8, spaceAfter=3)
BODY = ParagraphStyle("BODY", parent=styles["BodyText"], fontName="Times-Roman",
                      fontSize=11, leading=15.5, alignment=TA_JUSTIFY, spaceAfter=6)
BULLET = ParagraphStyle("BULLET", parent=BODY, leftIndent=16, bulletIndent=4, spaceAfter=3)
CAP = ParagraphStyle("CAP", parent=BODY, fontSize=9, textColor=GREY, alignment=1, spaceBefore=2)


def P(t, s=BODY):
    return Paragraph(t, s)


def bullets(items):
    return [Paragraph(f"• {it}", BULLET) for it in items]


def fig(name, max_w=15.5 * cm, max_h=20 * cm, caption=None):
    path = os.path.join(FIG, name)
    ir = ImageReader(path)
    iw, ih = ir.getSize()
    scale = min(max_w / iw, max_h / ih)
    w, h = iw * scale, ih * scale
    parts = [Spacer(1, 4), Image(path, width=w, height=h)]
    if caption:
        parts.append(P(caption, CAP))
    parts.append(Spacer(1, 6))
    return KeepTogether(parts)


def metric_table(rows, col_widths):
    t = Table(rows, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9.5),
        ("FONT", (0, 1), (-1, -1), "Helvetica", 9.5),
        ("BACKGROUND", (0, 0), (-1, 0), BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BBBBBB")),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F6FB")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


story = []
A = story.append

# ============================ 8.7 METHODOLOGIE JUSTIFIEE ============================
A(P("8.7&nbsp;&nbsp;&nbsp;Transfer Learning &ndash; M&eacute;thodologie et justification des choix", H1))
A(P("Cette section d&eacute;taille et <b>justifie</b> chaque choix de param&egrave;tre&nbsp;: d&eacute;coupage des "
    "donn&eacute;es, pr&eacute;traitement, augmentation, architecture de la t&ecirc;te, gestion du d&eacute;s&eacute;quilibre et "
    "protocole de fine-tuning."))

A(P("a)&nbsp;&nbsp;Pourquoi le Transfer Learning", H2))
A(P("Le CNN entra&icirc;n&eacute; de z&eacute;ro (8.3) g&eacute;n&eacute;ralise mal (accuracy de validation &asymp;&nbsp;50&nbsp;%, "
    "macro F1-score 0,20)&nbsp;: avec un jeu limit&eacute;, il pr&eacute;dit massivement la classe majoritaire "
    "<i>Normal</i>. Le Transfer Learning r&eacute;utilise des r&eacute;seaux pr&eacute;-entra&icirc;n&eacute;s sur ImageNet "
    "(1,2&nbsp;million d&rsquo;images)&nbsp;: leurs premi&egrave;res couches d&eacute;tectent d&eacute;j&agrave; des motifs "
    "g&eacute;n&eacute;riques (contours, textures) qu&rsquo;il suffit d&rsquo;adapter &agrave; la radiographie. C&rsquo;est "
    "l&rsquo;&eacute;tat de l&rsquo;art pour la classification d&rsquo;images m&eacute;dicales avec peu de donn&eacute;es."))

A(P("b)&nbsp;&nbsp;Reproductibilit&eacute; (graine al&eacute;atoire 42)", H2))
A(P("La graine est fix&eacute;e &agrave; 42 pour NumPy, TensorFlow et les g&eacute;n&eacute;rateurs d&rsquo;images. Cons&eacute;quence&nbsp;: "
    "m&ecirc;me d&eacute;coupage, m&ecirc;me ordre de m&eacute;lange et m&ecirc;mes tirages d&rsquo;augmentation &agrave; chaque ex&eacute;cution. "
    "C&rsquo;est indispensable pour comparer les mod&egrave;les &agrave; armes &eacute;gales et pour pouvoir reproduire et "
    "d&eacute;fendre exactement les chiffres pr&eacute;sent&eacute;s."))

A(P("c)&nbsp;&nbsp;Pourquoi un d&eacute;coupage <i>stratifi&eacute;</i> 70&nbsp;/&nbsp;15&nbsp;/&nbsp;15", H2))
A(P("<b>Pourquoi stratifier&nbsp;?</b> Le jeu est fortement d&eacute;s&eacute;quilibr&eacute; (Normal 48&nbsp;%, Viral "
    "Pneumonia 6&nbsp;%). Un d&eacute;coupage purement al&eacute;atoire risquerait de sous- ou sur-repr&eacute;senter une "
    "classe rare dans le test (par ex. trop peu de Viral Pneumonia), rendant la mesure instable et non "
    "repr&eacute;sentative. La <b>stratification</b> (param&egrave;tre <i>stratify</i> sur le label) garantit que "
    "<b>chaque</b> ensemble (train, validation, test) conserve les <b>m&ecirc;mes proportions de classes</b> que "
    "le jeu complet."))
A(P("<b>Pourquoi 70&nbsp;/&nbsp;15&nbsp;/&nbsp;15&nbsp;?</b> 70&nbsp;% pour l&rsquo;entra&icirc;nement (assez de donn&eacute;es pour "
    "fine-tuner)&nbsp;; 15&nbsp;% de validation, servant au r&eacute;glage et aux callbacks (EarlyStopping, "
    "ReduceLR) sans jamais intervenir dans la mesure finale&nbsp;; 15&nbsp;% de test, <b>jamais vus</b>, pour "
    "l&rsquo;&eacute;valuation finale. Le d&eacute;coupage se fait en deux temps (70&nbsp;% / 30&nbsp;%, puis ce 30&nbsp;% "
    "coup&eacute; en deux), avec <i>random_state&nbsp;=&nbsp;42</i> (d&eacute;terministe). La s&eacute;paration porte sur les "
    "images&nbsp;: une image ne peut appartenir qu&rsquo;&agrave; un seul ensemble &rarr; <b>pas de fuite de "
    "donn&eacute;es</b>."))
A(metric_table(
    [["Classe", "Train", "Validation", "Test"],
     ["COVID", "2 531", "543", "542"],
     ["Lung Opacity", "4 208", "902", "902"],
     ["Normal", "7 134", "1 529", "1 529"],
     ["Viral Pneumonia", "942", "201", "202"],
     ["Total", "14 815", "3 175", "3 175"]],
    [4.6 * cm, 3 * cm, 3 * cm, 3 * cm]))

A(P("d)&nbsp;&nbsp;Taille d&rsquo;image 224&times;224 et conversion RGB", H2))
A(P("Les images sont redimensionn&eacute;es en <b>224&times;224</b> car ResNet50 et EfficientNetB0 ont &eacute;t&eacute; "
    "pr&eacute;-entra&icirc;n&eacute;s &agrave; cette r&eacute;solution&nbsp;; la conserver permet de r&eacute;utiliser pleinement les "
    "poids ImageNet. Les radiographies &eacute;tant en niveaux de gris, elles sont charg&eacute;es en <b>3 canaux "
    "(RGB)</b> par r&eacute;plication du canal&nbsp;: les poids ImageNet attendent 3 canaux. On ne fabrique pas "
    "de couleur, on respecte le format d&rsquo;entr&eacute;e."))

A(P("e)&nbsp;&nbsp;Normalisation par <i>preprocess_input</i> d&eacute;di&eacute;", H2))
A(P("Chaque backbone applique <b>sa propre</b> normalisation (<i>resnet50.preprocess_input</i> &ne; "
    "<i>efficientnet.preprocess_input</i>), celle attendue par ses poids pr&eacute;-entra&icirc;n&eacute;s, et non un "
    "simple <i>/255</i> g&eacute;n&eacute;rique&nbsp;: sinon les statistiques d&rsquo;entr&eacute;e ne correspondraient pas &agrave; "
    "celles vues pendant l&rsquo;entra&icirc;nement ImageNet, d&eacute;gradant les performances."))

A(P("f)&nbsp;&nbsp;Augmentation de donn&eacute;es (train uniquement) &ndash; valeurs justifi&eacute;es", H2))
A(P("L&rsquo;augmentation n&rsquo;est appliqu&eacute;e qu&rsquo;au <b>train</b>, pour accro&icirc;tre artificiellement sa "
    "diversit&eacute; et limiter le sur-apprentissage&nbsp;; validation et test restent intacts (repr&eacute;sentatifs "
    "du r&eacute;el). Chaque valeur est choisie pour rester <b>r&eacute;aliste en radiologie</b>&nbsp;:"))
for b in bullets([
    "<b>rotation &plusmn;10&deg;</b> : un patient n&rsquo;est jamais parfaitement align&eacute;, mais une grande rotation serait irr&eacute;aliste sur un thorax.",
    "<b>d&eacute;calages &plusmn;8&nbsp;%</b> (horizontal et vertical) : tol&eacute;rance au cadrage.",
    "<b>zoom &plusmn;10&nbsp;%</b> : variation de distance / cadrage.",
    "<b>retournement horizontal</b> : un thorax reste plausible en miroir gauche/droite.",
    "<b>pas de retournement vertical</b> : une radio retourn&eacute;e (c&oelig;ur en haut) n&rsquo;a aucun sens anatomique et n&rsquo;introduirait que du bruit.",
    "<b>fill_mode = nearest</b> : les pixels cr&eacute;&eacute;s par rotation/d&eacute;calage sont combl&eacute;s par le voisin le plus proche, &eacute;vitant des bandes noires artificielles.",
]):
    A(b)

A(P("g)&nbsp;&nbsp;Architecture de la t&ecirc;te de classification", H2))
A(P("Sur le backbone gel&eacute;, on ajoute&nbsp;: <i>GlobalAveragePooling2D &rarr; BatchNormalization &rarr; "
    "Dropout(0,4) &rarr; Dense(128, ReLU) &rarr; Dropout(0,3) &rarr; Dense(4, Softmax)</i>. Le "
    "<b>GlobalAveragePooling</b> r&eacute;sume les cartes de caract&eacute;ristiques en un vecteur compact (moins de "
    "param&egrave;tres qu&rsquo;un Flatten, donc moins de sur-apprentissage). Les deux <b>Dropout</b> (0,4 puis "
    "0,3) et la <b>BatchNormalization</b> r&eacute;gularisent et stabilisent l&rsquo;apprentissage. La sortie "
    "<b>Softmax &agrave; 4 neurones</b> donne une probabilit&eacute; par classe."))

A(P("h)&nbsp;&nbsp;Taille de lot (batch&nbsp;=&nbsp;32)", H2))
A(P("Compromis classique&nbsp;: assez grand pour des gradients stables et une bonne utilisation du "
    "mat&eacute;riel, assez petit pour tenir en m&eacute;moire. 32 est une valeur standard et robuste."))

A(P("i)&nbsp;&nbsp;Gestion du d&eacute;s&eacute;quilibre &ndash; poids de classe &laquo;&nbsp;balanced&nbsp;&raquo;", H2))
A(P("Des poids de classe sont calcul&eacute;s avec <i>compute_class_weight(\"balanced\")</i> sur le "
    "<b>train uniquement</b> (anti-fuite)&nbsp;: chaque classe re&ccedil;oit un poids inversement proportionnel "
    "&agrave; sa fr&eacute;quence. Le mod&egrave;le est donc <b>davantage p&eacute;nalis&eacute;</b> lorsqu&rsquo;il se trompe sur une "
    "classe rare, ce qui l&rsquo;emp&ecirc;che de &laquo;&nbsp;tout pr&eacute;dire Normal&nbsp;&raquo;."))
A(metric_table(
    [["Classe", "Poids"], ["COVID", "1,463"], ["Lung Opacity", "0,880"],
     ["Normal", "0,519"], ["Viral Pneumonia", "3,932"]],
    [6 * cm, 3 * cm]))

A(P("j)&nbsp;&nbsp;Fine-tuning en deux phases &ndash; pourquoi ce protocole", H2))
for b in bullets([
    "<b>Phase 1 (t&ecirc;te seule, 10 &eacute;poques, lr&nbsp;=&nbsp;1e-4)</b> : le backbone est gel&eacute; (&asymp;&nbsp;267&nbsp;000 param&egrave;tres entra&icirc;nables sur 23,8&nbsp;millions pour ResNet50). On laisse la nouvelle t&ecirc;te se stabiliser sans d&eacute;truire les poids ImageNet par des gradients initiaux trop grands.",
    "<b>Phase 2 (fine-tuning, 15 &eacute;poques, lr&nbsp;=&nbsp;1e-5)</b> : on <b>d&eacute;g&egrave;le les 30 derni&egrave;res couches</b> du backbone (les plus sp&eacute;cifiques), les premi&egrave;res couches restant gel&eacute;es. On <b>recompile</b> avec un taux d&rsquo;apprentissage <b>10&times; plus faible</b> (1e-5) pour ajuster finement ces couches au domaine m&eacute;dical <b>sans &laquo;&nbsp;casser&nbsp;&raquo;</b> les poids pr&eacute;-entra&icirc;n&eacute;s.",
]):
    A(b)

A(P("k)&nbsp;&nbsp;Callbacks", H2))
for b in bullets([
    "<b>EarlyStopping</b> (patience 5, restauration des meilleurs poids) : arr&ecirc;te l&rsquo;entra&icirc;nement si la validation ne progresse plus &rarr; &eacute;vite le sur-apprentissage et conserve le meilleur &eacute;tat.",
    "<b>ModelCheckpoint</b> : sauvegarde automatiquement le meilleur mod&egrave;le (sur <i>val_loss</i>).",
    "<b>ReduceLROnPlateau</b> (facteur 0,3, patience 2) : r&eacute;duit le taux d&rsquo;apprentissage quand la validation stagne, pour affiner la convergence.",
]):
    A(b)

A(P("l)&nbsp;&nbsp;M&eacute;trique d&rsquo;&eacute;valuation", H2))
A(P("Vu le d&eacute;s&eacute;quilibre, l&rsquo;accuracy est trompeuse (un classifieur &laquo;&nbsp;tout Normal&nbsp;&raquo; "
    "atteindrait 48&nbsp;%). La m&eacute;trique de s&eacute;lection est donc le <b>macro F1-score</b> (moyenne non "
    "pond&eacute;r&eacute;e des F1 par classe&nbsp;: chaque classe p&egrave;se autant), compl&eacute;t&eacute;e par le rappel par "
    "classe (notamment COVID) comme garde-fou clinique."))

# ============================ 8.8 RESNET50 (RETENU) ============================
A(P("8.8&nbsp;&nbsp;&nbsp;Transfer Learning &ndash; ResNet50 (mod&egrave;le retenu)", H1))
A(P("&Eacute;valuation sur le test fig&eacute; (3&nbsp;175 images). ResNet50 fine-tun&eacute; obtient les meilleures "
    "performances de tous les mod&egrave;les test&eacute;s&nbsp;: c&rsquo;est le mod&egrave;le retenu."))
A(P("Performance globale", H3))
for b in bullets(["Accuracy&nbsp;: <b>0,933</b>", "Macro F1-score&nbsp;: <b>0,936</b>",
                  "Pr&eacute;cision macro&nbsp;: 0,941", "Rappel macro&nbsp;: 0,932",
                  "Weighted F1-score&nbsp;: 0,933"]):
    A(b)
A(P("Classification report", H3))
A(metric_table(
    [["Classe", "Précision", "Rappel", "F1-score", "Support"],
     ["COVID", "0,965", "0,958", "0,961", "542"],
     ["Lung Opacity", "0,943", "0,873", "0,906", "902"],
     ["Normal", "0,917", "0,960", "0,938", "1 529"],
     ["Viral Pneumonia", "0,940", "0,936", "0,938", "202"],
     ["Macro avg", "0,941", "0,932", "0,936", "3 175"]],
    [4.2 * cm, 2.6 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm]))
A(P("<b>Observations.</b> ResNet50 fine-tun&eacute; atteint un macro F1-score de 0,936, le meilleur de "
    "l&rsquo;&eacute;tude, avec un excellent &eacute;quilibre sur les quatre classes&nbsp;: rappel COVID 0,958 "
    "(garde-fou clinique), et la classe difficile <i>Lung Opacity</i> est la mieux trait&eacute;e de tous les "
    "mod&egrave;les (rappel 0,873). C&rsquo;est pourquoi il est retenu comme mod&egrave;le final."))

# ============================ 8.9 EFFICIENTNET ============================
A(P("8.9&nbsp;&nbsp;&nbsp;Transfer Learning &ndash; EfficientNetB0", H1))
A(P("&Eacute;valuation sur le test fig&eacute; (3&nbsp;175 images)."))
A(P("Performance globale", H3))
for b in bullets(["Accuracy&nbsp;: <b>0,893</b>", "Macro F1-score&nbsp;: <b>0,896</b>",
                  "Pr&eacute;cision macro&nbsp;: 0,881", "Rappel macro&nbsp;: 0,913",
                  "Weighted F1-score&nbsp;: 0,893"]):
    A(b)
A(P("Classification report", H3))
A(metric_table(
    [["Classe", "Précision", "Rappel", "F1-score", "Support"],
     ["COVID", "0,836", "0,924", "0,878", "542"],
     ["Lung Opacity", "0,860", "0,900", "0,880", "902"],
     ["Normal", "0,939", "0,869", "0,902", "1 529"],
     ["Viral Pneumonia", "0,890", "0,960", "0,924", "202"],
     ["Macro avg", "0,881", "0,913", "0,896", "3 175"]],
    [4.2 * cm, 2.6 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm]))
A(P("<b>Observations.</b> EfficientNetB0 fine-tun&eacute; atteint lui aussi un tr&egrave;s bon niveau (macro F1 "
    "0,896), avec un bon rappel macro (0,913) et un rappel COVID de 0,924. Il reste l&eacute;g&egrave;rement en "
    "de&ccedil;&agrave; de ResNet50 sur le F1-macro global, ce qui le place en second mod&egrave;le de l&rsquo;&eacute;tude."))

# ============================ 8.10 COMPARAISON ============================
A(P("8.10&nbsp;&nbsp;&nbsp;Comparaison des mod&egrave;les", H1))
A(P("Les deux mod&egrave;les de Transfer Learning sont &eacute;valu&eacute;s sur le test fig&eacute; (3&nbsp;175 images)&nbsp;; "
    "le CNN seul (8.4) et l&rsquo;hybride CNN&nbsp;+&nbsp;Gradient Boosting (8.6) ont &eacute;t&eacute; &eacute;valu&eacute;s sur le "
    "d&eacute;coupage 80/20 et sont rappel&eacute;s &agrave; titre indicatif."))
A(metric_table(
    [["Modèle", "Macro F1", "Accuracy", "Rappel COVID", "Test"],
     ["ResNet50 (retenu)", "0,936", "0,933", "0,958", "figé · 3 175"],
     ["EfficientNetB0", "0,896", "0,893", "0,924", "figé · 3 175"],
     ["Hybride CNN + GB (8.6)", "0,83", "0,83", "0,80", "80/20"],
     ["CNN seul (8.4)", "0,20", "0,50", "0,00", "80/20"]],
    [4.8 * cm, 2.3 * cm, 2.3 * cm, 2.6 * cm, 2.7 * cm]))
A(P("<b>Lecture.</b> Le Transfer Learning am&eacute;liore tr&egrave;s nettement la classification par rapport au "
    "CNN entra&icirc;n&eacute; de z&eacute;ro (macro F1 0,20, qui ne d&eacute;tecte aucun COVID) et &agrave; l&rsquo;approche hybride "
    "(0,83). <b>ResNet50 fine-tun&eacute; est le meilleur mod&egrave;le (macro F1 0,936)</b> et constitue le "
    "mod&egrave;le retenu, suivi de pr&egrave;s par EfficientNetB0 (0,896). Cela valide l&rsquo;apport des "
    "architectures pr&eacute;-entra&icirc;n&eacute;es sur ImageNet pour ce probl&egrave;me."))

# ============================ 8.11 INTERPRETABILITE ============================
A(P("8.11&nbsp;&nbsp;&nbsp;Interpr&eacute;tabilit&eacute; &eacute;tendue &ndash; Grad-CAM (Transfer Learning) + SHAP", H1))
A(P("La section 8.5 appliquait Grad-CAM au CNN. L&rsquo;interpr&eacute;tabilit&eacute; est ici &eacute;tendue aux deux "
    "mod&egrave;les de Transfer Learning et compl&eacute;t&eacute;e par SHAP."))
A(P("a)&nbsp;&nbsp;Grad-CAM sur ResNet50 et EfficientNetB0", H2))
A(P("Grad-CAM est appliqu&eacute; aux deux mod&egrave;les, sur la derni&egrave;re couche convolutive&nbsp;: un exemple "
    "correctement class&eacute; par classe (image originale | carte de chaleur | superposition) et des cas mal "
    "class&eacute;s, pour observer o&ugrave; le mod&egrave;le regarde lorsqu&rsquo;il se trompe."))
A(fig("gradcam_resnet50_COVID.png", caption="Grad-CAM &ndash; ResNet50, cas COVID correctement class&eacute;."))
A(fig("gradcam_efficientnetb0_COVID.png", caption="Grad-CAM &ndash; EfficientNetB0, cas COVID correctement class&eacute;."))
A(P("b)&nbsp;&nbsp;SHAP", H2))
A(P("shap.GradientExplainer (robuste sur les mod&egrave;les fonctionnels Keras r&eacute;cents) est appliqu&eacute; &agrave; "
    "un mod&egrave;le de Transfer Learning&nbsp;: fond d&rsquo;environ 24 images d&rsquo;entra&icirc;nement, explication de "
    "8 images de test (2 par classe). Cette approche compl&egrave;te Grad-CAM en attribuant &agrave; chaque pixel "
    "une contribution (positive ou n&eacute;gative) &agrave; la pr&eacute;diction de la classe. Usage qualitatif."))
A(fig("shap_efficientnetb0.png",
      caption="Valeurs SHAP par classe (EfficientNetB0). &Agrave; gauche la radio originale&nbsp;; pour "
              "chaque classe, les pixels en rouge poussent vers cette classe, en bleu l&rsquo;en &eacute;loignent."))
A(P("c)&nbsp;&nbsp;Lecture et garde-fou", H2))
A(P("Une activation localis&eacute;e sur les champs pulmonaires conforte la plausibilit&eacute; clinique de la "
    "pr&eacute;diction. &Agrave; l&rsquo;inverse, une activation concentr&eacute;e sur les bords, les marqueurs ou les "
    "annotations signalerait un biais d&rsquo;apprentissage de raccourci (<i>shortcut learning</i>)&nbsp;: le "
    "mod&egrave;le exploiterait des artefacts corr&eacute;l&eacute;s &agrave; la source de l&rsquo;image plut&ocirc;t que la "
    "pathologie. Les classes provenant de bases diff&eacute;rentes, ce risque est r&eacute;el et "
    "l&rsquo;interpr&eacute;tabilit&eacute; sert pr&eacute;cis&eacute;ment &agrave; l&rsquo;auditer."))

# ============================ 8.12 CONCLUSION MODELISATION ============================
A(P("8.12&nbsp;&nbsp;&nbsp;Conclusion de la mod&eacute;lisation", H1))
A(P("La progression Baseline ML &rarr; CNN &rarr; Gradient Boosting &rarr; Transfer Learning montre un "
    "gain net et coh&eacute;rent. <b>Le mod&egrave;le retenu est ResNet50 fine-tun&eacute;</b> (macro F1 0,936, "
    "accuracy 0,933, rappel COVID 0,958), qui obtient les meilleures performances de l&rsquo;&eacute;tude&nbsp;; "
    "EfficientNetB0 (0,896) le suit de pr&egrave;s. Le d&eacute;coupage stratifi&eacute; anti-fuite, les poids de "
    "classe, le fine-tuning en deux phases et l&rsquo;audit Grad-CAM/SHAP garantissent une &eacute;valuation "
    "honn&ecirc;te et d&eacute;fendable."))


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Roman", 10)
    canvas.drawRightString(A4[0] - 2 * cm, 1.3 * cm, str(52 + doc.page))
    canvas.restoreState()


doc = SimpleDocTemplate(
    NEWPAGES, pagesize=A4,
    leftMargin=2.3 * cm, rightMargin=2.3 * cm, topMargin=2.2 * cm, bottomMargin=2.0 * cm,
)
doc.build(story, onFirstPage=footer, onLaterPages=footer)
n_new = len(PdfReader(NEWPAGES).pages)
print(f"Pages generees : {n_new}")

orig = PdfReader(ORIG)
new = PdfReader(NEWPAGES)
w = PdfWriter()
INSERT_AFTER = 53  # apres la matrice de confusion du Boosting (fin de 8.6)
for p in orig.pages[:INSERT_AFTER]:
    w.add_page(p)
for p in new.pages:
    w.add_page(p)
for p in orig.pages[INSERT_AFTER:]:
    w.add_page(p)
with open(OUT, "wb") as f:
    w.write(f)
os.remove(NEWPAGES)
print(f"PDF mis a jour : {OUT}  | total pages : {len(PdfReader(OUT).pages)}")

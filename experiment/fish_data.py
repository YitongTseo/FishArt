"""
Curated, stratified dataset of tropical/aquarium fish species with a
COMPONENTISED expert mate-choice score (auditable, not a black box).

Each species is scored on four behavioural sub-axes, 0-3 each, from
mating-system biology ONLY (never coloration -> avoids circularity):

  fert   fertilisation / egg mode
         0 broadcast or group spawn (gametes en masse)
         1 pelagic pair spawn
         2 demersal / nest / substrate spawn (male tends)
         3 internal fertilisation (livebearer), mouthbrooder, or male brood
  court  courtship elaboration
         0 none   1 simple chase   2 elaborate display   3 spectacular
                                        (lek dance / bower / nuptial flashing)
  system mating system
         0 promiscuous mass spawn   1 monogamous pair
         2 haremic                  3 polygynous lek / display arena
  choice strength of mate choice
         0 negligible   1 mutual / weak   2 female choice
         3 textbook strong female choice (or male choice if sex-role reversed)

mate_choice_index = 10 * (fert+court+system+choice) / 12   (0-10)

Deliberately includes "trap" cases (bright but low-choice; drab but high-choice)
so the test cannot be rigged.
"""

# sci, common, family, habitat, fert, court, system, choice, note
_RAW = [
    # --- broadcast / group spawners: minimal individual choice ---
    ("Sardinella longiceps","Indian oil sardine","Clupeidae","marine",0,0,0,0,"pelagic mass spawn, silver"),
    ("Katsuwonus pelamis","Skipjack tuna","Scombridae","marine",0,0,0,0,"pelagic broadcast"),
    ("Lutjanus campechanus","Red snapper","Lutjanidae","marine",0,0,0,0,"aggregation spawn"),
    ("Epinephelus striatus","Nassau grouper","Serranidae","marine",0,1,0,0,"huge spawning aggregations"),
    ("Acanthurus coeruleus","Atlantic blue tang","Acanthuridae","marine",0,0,0,1,"group spawn"),
    ("Paracanthurus hepatus","Blue tang","Acanthuridae","marine",0,1,0,1,"TRAP: brilliant blue, aggregation spawner"),
    ("Zebrasoma flavescens","Yellow tang","Acanthuridae","marine",0,1,0,1,"TRAP: vivid yellow, aggregation spawner"),
    ("Mulloidichthys vanicolensis","Yellowfin goatfish","Mullidae","marine",0,0,0,1,"group spawn"),
    ("Paracheirodon axelrodi","Cardinal tetra","Characidae","freshwater",1,1,0,1,"TRAP: neon but egg-scatterer, low choice"),
    ("Paracheirodon innesi","Neon tetra","Characidae","freshwater",1,1,0,1,"TRAP: neon egg-scatterer"),
    ("Danio rerio","Zebrafish","Cyprinidae","freshwater",1,1,0,1,"scatter spawner, some chasing"),
    ("Puntigrus tetrazona","Tiger barb","Cyprinidae","freshwater",1,1,0,1,"egg scatterer"),
    ("Trigonostigma heteromorpha","Harlequin rasbora","Cyprinidae","freshwater",1,1,0,1,"plant spawner"),

    # --- pelagic pair / mobile spawners ---
    ("Zanclus cornutus","Moorish idol","Zanclidae","marine",1,1,1,1,"pelagic pair spawn"),
    ("Pterois volitans","Red lionfish","Scorpaenidae","marine",1,1,2,1,"TRAP: striking = aposematic warning, not display"),

    # --- monogamous pairs / nest tenders: real but bounded choice ---
    ("Chaetodon auriga","Threadfin butterflyfish","Chaetodontidae","marine",1,1,1,2,"long-term monogamous pairs"),
    ("Chaetodon lunula","Raccoon butterflyfish","Chaetodontidae","marine",1,1,1,2,"monogamous pairs"),
    ("Amphiprion ocellaris","Clown anemonefish","Pomacentridae","marine",2,1,1,1,"monogamous; orange partly symbiotic"),
    ("Amphiprion percula","Orange clownfish","Pomacentridae","marine",2,1,1,1,"monogamous pair"),
    ("Chromis viridis","Blue-green chromis","Pomacentridae","marine",2,1,0,2,"male nest, females choose"),
    ("Dascyllus aruanus","Humbug dascyllus","Pomacentridae","marine",2,1,1,2,"nest-tending damselfish"),
    ("Chrysiptera cyanea","Sapphire devil","Pomacentridae","marine",2,1,1,2,"male nest, female choice"),
    ("Elacatinus oceanops","Neon goby","Gobiidae","marine",2,1,1,1,"pair, male tends eggs"),
    ("Ecsenius bicolor","Bicolor blenny","Blenniidae","marine",2,1,2,2,"male nest, female choice"),
    ("Pterapogon kauderni","Banggai cardinalfish","Apogonidae","marine",3,1,1,2,"paternal mouthbrooder, pair"),
    ("Balistoides conspicillum","Clown triggerfish","Balistidae","marine",2,1,2,1,"TRAP: bold pattern, haremic nester"),
    ("Rhinecanthus aculeatus","Picasso triggerfish","Balistidae","marine",2,1,2,1,"nest, haremic"),
    ("Ostracion cubicus","Yellow boxfish","Ostraciidae","marine",1,2,2,2,"haremic, male display"),
    ("Oxycirrhites typus","Longnose hawkfish","Cirrhitidae","marine",1,1,2,1,"haremic"),

    # --- haremic / sex-changing: male display, female choice ---
    ("Pomacanthus imperator","Emperor angelfish","Pomacanthidae","marine",1,2,2,2,"haremic male display at dusk"),
    ("Centropyge bicolor","Bicolor angelfish","Pomacanthidae","marine",1,2,2,2,"haremic, protogynous"),
    ("Thalassoma bifasciatum","Bluehead wrasse","Labridae","marine",1,2,3,2,"terminal males defend, females choose"),
    ("Labroides dimidiatus","Cleaner wrasse","Labridae","marine",1,1,2,1,"haremic, protogynous"),
    ("Scarus quoyi","Quoy's parrotfish","Scaridae","marine",1,2,2,2,"harem, terminal-male display"),
    ("Pseudanthias squamipinnis","Sea goldie (anthias)","Serranidae","marine",1,2,3,2,"haremic, protogynous, male courts"),
    ("Cyphotilapia frontosa","Frontosa cichlid","Cichlidae","freshwater",3,1,2,1,"polygynous mouthbrooder"),

    # --- lek / display / nest builders / livebearers: strong choice ---
    ("Symphysodon aequifasciatus","Discus","Cichlidae","freshwater",2,2,1,2,"monogamous biparental, mutual choice"),
    ("Pterophyllum scalare","Freshwater angelfish","Cichlidae","freshwater",2,1,1,2,"monogamous pair"),
    ("Mikrogeophagus ramirezi","Ram cichlid","Cichlidae","freshwater",2,2,1,2,"monogamous, courts"),
    ("Apistogramma cacatuoides","Cockatoo dwarf cichlid","Cichlidae","freshwater",2,2,2,2,"harem, male display"),
    ("Neolamprologus brichardi","Fairy cichlid","Cichlidae","freshwater",2,1,1,1,"monogamous cooperative"),
    ("Tropheus moorii","Blunthead cichlid","Cichlidae","freshwater",3,2,2,2,"mouthbrooder, male display"),
    ("Maylandia zebra","Zebra mbuna","Cichlidae","freshwater",3,2,3,2,"male territory display, mouthbrooder"),
    ("Pseudotropheus socolofi","Powder-blue mbuna","Cichlidae","freshwater",3,2,3,2,"male display, mouthbrooder"),
    ("Aulonocara stuartgranti","Peacock cichlid","Cichlidae","freshwater",3,3,3,3,"lek-like male display, female choice"),
    ("Melanotaenia boesemani","Boeseman's rainbowfish","Melanotaeniidae","freshwater",1,2,2,2,"male courtship display"),
    ("Melanotaenia praecox","Dwarf neon rainbowfish","Melanotaeniidae","freshwater",1,2,2,2,"male display"),
    ("Trichogaster lalius","Dwarf gourami","Osphronemidae","freshwater",2,2,2,2,"bubble nest, courts"),
    ("Trichopodus trichopterus","Three-spot gourami","Osphronemidae","freshwater",2,1,2,1,"bubble nest"),
    ("Macropodus opercularis","Paradise fish","Osphronemidae","freshwater",2,2,2,2,"bubble nest, display"),
    ("Betta splendens","Siamese fighting fish","Osphronemidae","freshwater",2,3,3,2,"bubble nest + flaring display"),
    ("Gasterosteus aculeatus","Three-spined stickleback","Gasterosteidae","temperate-fw",2,3,2,3,"essay ex; red belly, nest, dance"),
    ("Poecilia latipinna","Sailfin molly","Poeciliidae","freshwater",3,2,3,2,"livebearer, male sail display"),
    ("Xiphophorus maculatus","Southern platyfish","Poeciliidae","freshwater",3,1,2,2,"livebearer, female choice"),
    ("Xiphophorus hellerii","Green swordtail","Poeciliidae","freshwater",3,2,3,3,"female choice on sword length"),
    ("Poecilia reticulata","Guppy","Poeciliidae","freshwater",3,3,3,3,"THE model of female mate choice"),
    ("Poecilia wingei","Endler's livebearer","Poeciliidae","freshwater",3,3,3,3,"strong female choice, male display"),
    ("Cirrhilabrus solorensis","Solor fairy wrasse","Labridae","marine",1,3,3,2,"male nuptial flashing"),
    ("Paracheilinus carpenteri","Carpenter's flasher wrasse","Labridae","marine",1,3,3,3,"males flash intensified nuptial colours"),
    ("Synchiropus splendidus","Mandarinfish","Callionymidae","marine",1,3,3,3,"spectacular dusk courtship, female picks"),
    ("Synchiropus picturatus","Psychedelic mandarin","Callionymidae","marine",1,3,3,3,"elaborate courtship rise"),
    ("Hippocampus kuda","Common seahorse","Syngnathidae","marine",3,3,1,2,"SEX-ROLE REVERSED: male brood, courtship dance"),
    ("Torquigener albomaculosus","White-spotted pufferfish","Tetraodontidae","marine",2,3,3,3,"TRAP: drab body, 'art' is external sand mandala"),

    # ===== iteration 3 additions (toward ~150 species) =====
    # -- silvery pelagic / broadcast: low-choice, low-colour anchors --
    ("Rastrelliger kanagurta","Indian mackerel","Scombridae","marine",0,0,0,0,"pelagic broadcast"),
    ("Selar crumenophthalmus","Bigeye scad","Carangidae","marine",0,0,0,0,"pelagic broadcast"),
    ("Caranx ignobilis","Giant trevally","Carangidae","marine",0,0,0,0,"pelagic broadcast"),
    ("Mugil cephalus","Flathead mullet","Mugilidae","marine",0,0,0,0,"broadcast spawn"),
    ("Chanos chanos","Milkfish","Chanidae","marine",0,0,0,0,"broadcast spawn"),
    ("Sphyraena barracuda","Great barracuda","Sphyraenidae","marine",0,0,0,0,"pelagic"),
    ("Sardinops sagax","Pacific sardine","Clupeidae","marine",0,0,0,0,"broadcast"),
    ("Lutjanus kasmira","Bluestripe snapper","Lutjanidae","marine",0,0,0,1,"TRAP: yellow, aggregation spawn"),
    ("Haemulon sciurus","Bluestriped grunt","Haemulidae","marine",0,0,0,1,"aggregation"),
    # -- surgeonfish / rabbitfish: bright low-choice traps --
    ("Acanthurus leucosternon","Powder blue tang","Acanthuridae","marine",0,1,0,1,"TRAP: gorgeous, aggregation spawner"),
    ("Zebrasoma xanthurum","Purple tang","Acanthuridae","marine",0,1,0,1,"TRAP: vivid, aggregation spawner"),
    ("Zebrasoma veliferum","Sailfin tang","Acanthuridae","marine",0,1,0,1,"aggregation spawner"),
    ("Naso lituratus","Orangespine unicornfish","Acanthuridae","marine",0,1,0,1,"group spawn"),
    ("Siganus vulpinus","Foxface rabbitfish","Siganidae","marine",1,1,1,1,"monogamous pairs"),
    # -- butterflyfish / bannerfish: monogamous, mid --
    ("Chaetodon semilarvatus","Masked butterflyfish","Chaetodontidae","marine",1,1,1,2,"monogamous pairs"),
    ("Heniochus acuminatus","Pennant bannerfish","Chaetodontidae","marine",1,1,1,2,"pairs"),
    ("Forcipiger flavissimus","Longnose butterflyfish","Chaetodontidae","marine",1,1,1,2,"pairs"),
    # -- marine angelfish: haremic, often dichromatic --
    ("Centropyge loricula","Flame angelfish","Pomacanthidae","marine",1,2,2,2,"haremic, protogynous"),
    ("Genicanthus melanospilos","Swallowtail angelfish","Pomacanthidae","marine",1,2,2,2,"haremic, sexually dichromatic"),
    ("Holacanthus ciliaris","Queen angelfish","Pomacanthidae","marine",1,2,2,2,"haremic display"),
    ("Pygoplites diacanthus","Regal angelfish","Pomacanthidae","marine",1,2,2,2,"haremic"),
    ("Pomacanthus paru","French angelfish","Pomacanthidae","marine",1,2,1,2,"monogamous pairs"),
    ("Holacanthus tricolor","Rock beauty","Pomacanthidae","marine",1,2,2,2,"haremic"),
    # -- wrasses: haremic display, some flashers --
    ("Coris gaimard","Yellowtail coris","Labridae","marine",1,2,2,2,"haremic, terminal-male display"),
    ("Gomphosus varius","Bird wrasse","Labridae","marine",1,2,2,2,"haremic display"),
    ("Macropharyngodon meleagris","Leopard wrasse","Labridae","marine",1,2,2,2,"haremic display"),
    ("Bodianus rufus","Spanish hogfish","Labridae","marine",1,2,2,2,"haremic"),
    ("Pseudocheilinus hexataenia","Sixline wrasse","Labridae","marine",1,1,2,1,"haremic"),
    ("Paracheilinus filamentosus","Filamented flasher wrasse","Labridae","marine",1,3,3,3,"male nuptial flashing"),
    ("Cirrhilabrus rubriventralis","Longfin fairy wrasse","Labridae","marine",1,3,3,2,"male nuptial display"),
    # -- parrotfish --
    ("Sparisoma viride","Stoplight parrotfish","Scaridae","marine",1,2,2,2,"haremic, terminal-male display"),
    ("Chlorurus sordidus","Bullethead parrotfish","Scaridae","marine",1,2,2,2,"haremic"),
    # -- damsels / clownfish --
    ("Premnas biaculeatus","Maroon clownfish","Pomacentridae","marine",2,1,1,1,"monogamous"),
    ("Amphiprion frenatus","Tomato clownfish","Pomacentridae","marine",2,1,1,1,"monogamous"),
    ("Chromis cyanea","Blue chromis","Pomacentridae","marine",2,1,0,2,"male nest, female choice"),
    ("Stegastes partitus","Bicolor damselfish","Pomacentridae","marine",2,2,1,2,"male courtship sound + display"),
    # -- basslets / dottybacks / anthias / hamlets --
    ("Gramma loreto","Royal gramma","Grammatidae","marine",2,2,2,2,"male nest, haremic"),
    ("Pseudochromis fridmani","Orchid dottyback","Pseudochromidae","marine",2,2,2,2,"male nest, courts"),
    ("Pictichromis paccagnellae","Royal dottyback","Pseudochromidae","marine",2,2,2,2,"male nest"),
    ("Pseudanthias tuka","Purple anthias","Serranidae","marine",1,2,3,2,"haremic, male display"),
    ("Pseudanthias dispar","Peach anthias","Serranidae","marine",1,2,3,2,"haremic, male display"),
    ("Hypoplectrus puella","Barred hamlet","Serranidae","marine",1,2,1,2,"egg-trading hermaphrodite, colour morphs"),
    ("Serranus tigrinus","Harlequin bass","Serranidae","marine",1,2,1,2,"egg trader"),
    # -- gobies / firefish / jawfish / dragonets --
    ("Nemateleotris magnifica","Fire goby","Microdesmidae","marine",2,2,1,2,"pairs, fin display"),
    ("Nemateleotris decora","Elegant firefish","Microdesmidae","marine",2,2,1,2,"pairs, display"),
    ("Cryptocentrus cinctus","Yellow watchman goby","Gobiidae","marine",2,1,1,1,"pair, male tends"),
    ("Opistognathus aurifrons","Yellowhead jawfish","Opistognathidae","marine",3,1,1,2,"paternal mouthbrooder"),
    ("Synchiropus ocellatus","Scooter dragonet","Callionymidae","marine",1,2,3,2,"male courtship rise"),
    # -- boxfish / trigger / puffer: shape diversity --
    ("Ostracion meleagris","Spotted boxfish","Ostraciidae","marine",1,2,2,2,"haremic, sexually dichromatic"),
    ("Odonus niger","Redtooth triggerfish","Balistidae","marine",2,1,2,1,"nest, haremic"),
    ("Balistapus undulatus","Orange-lined triggerfish","Balistidae","marine",2,1,2,1,"nest, haremic"),
    ("Canthigaster valentini","Valentini puffer","Tetraodontidae","marine",2,2,2,2,"haremic, male courts"),
    # -- seahorses / pipefish: sex-role reversed --
    ("Hippocampus erectus","Lined seahorse","Syngnathidae","marine",3,3,1,2,"male brood, courtship dance"),
    ("Hippocampus reidi","Longsnout seahorse","Syngnathidae","marine",3,3,1,2,"male brood, courtship"),
    ("Doryrhamphus dactyliophorus","Banded pipefish","Syngnathidae","marine",3,2,1,2,"male brood"),
    # -- haplochromine / rift-lake cichlids: strong female choice --
    ("Astatotilapia burtoni","Burton's mouthbrooder","Cichlidae","freshwater",3,3,3,3,"classic sexual-selection model"),
    ("Pundamilia nyererei","Pundamilia","Cichlidae","freshwater",3,3,3,3,"female-choice speciation"),
    ("Sciaenochromis fryeri","Electric blue ahli","Cichlidae","freshwater",3,3,3,3,"male display"),
    ("Copadichromis borleyi","Redfin hap","Cichlidae","freshwater",3,3,3,3,"lek-like display"),
    ("Labidochromis caeruleus","Yellow lab","Cichlidae","freshwater",3,2,3,2,"male display, mouthbrooder"),
    ("Melanochromis auratus","Auratus cichlid","Cichlidae","freshwater",3,2,3,2,"strongly dichromatic"),
    ("Labeotropheus fuelleborni","Fuelleborni mbuna","Cichlidae","freshwater",3,2,3,2,"male display"),
    ("Nimbochromis venustus","Venustus","Cichlidae","freshwater",3,2,2,2,"mouthbrooder"),
    # -- Tanganyika / neotropical cichlids: varied --
    ("Julidochromis marlieri","Marlier's julie","Cichlidae","freshwater",2,1,1,1,"monogamous"),
    ("Lamprologus ocellatus","Shell-dweller cichlid","Cichlidae","freshwater",2,2,2,2,"shell nest, display"),
    ("Neolamprologus pulcher","Daffodil cichlid","Cichlidae","freshwater",2,1,1,1,"monogamous cooperative"),
    ("Amphilophus citrinellus","Midas cichlid","Cichlidae","freshwater",2,2,1,2,"monogamous"),
    ("Thorichthys meeki","Firemouth cichlid","Cichlidae","freshwater",2,2,1,2,"monogamous display"),
    ("Pelvicachromis pulcher","Kribensis","Cichlidae","freshwater",2,2,2,2,"cave, mutual (female colourful)"),
    ("Hemichromis bimaculatus","Jewel cichlid","Cichlidae","freshwater",2,2,1,2,"monogamous"),
    ("Andinoacara pulcher","Blue acara","Cichlidae","freshwater",2,2,1,2,"monogamous"),
    ("Geophagus altifrons","Eartheater cichlid","Cichlidae","freshwater",3,1,1,1,"mouthbrooder"),
    # -- killifish: brilliant males, strong choice --
    ("Nothobranchius rachovii","Rachow's nothobranch","Nothobranchiidae","freshwater",2,2,2,3,"brilliant males, female choice"),
    ("Nothobranchius furzeri","Turquoise killifish","Nothobranchiidae","freshwater",2,2,2,3,"male display"),
    ("Fundulopanchax gardneri","Blue lyretail killifish","Nothobranchiidae","freshwater",2,2,2,3,"male display"),
    ("Aphyosemion australe","Lyretail killifish","Nothobranchiidae","freshwater",2,2,2,3,"male display"),
    # -- more livebearers --
    ("Xiphophorus variatus","Variable platyfish","Poeciliidae","freshwater",3,1,2,2,"livebearer, female choice"),
    ("Poecilia sphenops","Common molly","Poeciliidae","freshwater",3,1,2,2,"livebearer"),
    ("Girardinus metallicus","Metallic livebearer","Poeciliidae","freshwater",3,2,2,2,"male display"),
    ("Gambusia affinis","Western mosquitofish","Poeciliidae","freshwater",3,1,1,1,"TRAP: internal fert but drab, coercive"),
    # -- rainbowfish / blue-eyes --
    ("Glossolepis incisus","Red rainbowfish","Melanotaeniidae","freshwater",1,2,2,2,"male red display"),
    ("Pseudomugil furcatus","Forktail blue-eye","Pseudomugilidae","freshwater",1,2,2,2,"male fin display"),
    # -- gouramis / anabantoids --
    ("Trichopodus leerii","Pearl gourami","Osphronemidae","freshwater",2,2,2,2,"bubble nest, display"),
    ("Trichogaster chuna","Honey gourami","Osphronemidae","freshwater",2,2,2,2,"bubble nest, display"),
    ("Sphaerichthys osphromenoides","Chocolate gourami","Osphronemidae","freshwater",3,1,1,1,"mouthbrooder"),
    ("Betta imbellis","Peaceful betta","Osphronemidae","freshwater",2,2,2,2,"nest, display"),
    # -- tetras: mostly low-choice scatter (bright traps) --
    ("Nematobrycon palmeri","Emperor tetra","Characidae","freshwater",1,2,1,2,"males display, some choice"),
    ("Hyphessobrycon pulchripinnis","Lemon tetra","Characidae","freshwater",1,1,0,1,"egg scatterer"),
    ("Hemigrammus erythrozonus","Glowlight tetra","Characidae","freshwater",1,1,0,1,"egg scatterer"),
    # -- barbs / danios --
    ("Pethia conchonius","Rosy barb","Cyprinidae","freshwater",1,1,0,1,"egg scatterer, males redden"),
    ("Danio margaritatus","Celestial pearl danio","Cyprinidae","freshwater",1,2,1,2,"males display"),
    # -- catfish / loach: shape diversity, mostly low ornament --
    ("Corydoras aeneus","Bronze cory","Callichthyidae","freshwater",1,1,0,1,"T-position spawner"),
    ("Ancistrus cirrhosus","Bristlenose pleco","Loricariidae","freshwater",2,2,2,2,"cave, male bristles ornament"),
]

def _index(fert, court, system, choice):
    return round(10.0 * (fert + court + system + choice) / 12.0, 2)

# Discrete mating-BEHAVIOUR type, derived from reproductive mode (fert) with
# fert==3 split by the note. Ordered roughly random -> choice for the legend.
MODE_ORDER = ["Broadcast / group spawn", "Pelagic pair spawn", "Nest builder / guarder",
              "Harem / sex-changer", "Mouthbrooder", "Male brooder", "Livebearer (internal)"]

def mating_mode(fert, court, system, note, family=""):
    n = note.lower()
    if fert == 0:
        return "Broadcast / group spawn"
    if fert == 3:
        if family == "Poeciliidae":
            return "Livebearer (internal)"
        if family == "Syngnathidae":
            return "Male brooder"
        if family == "Cichlidae" or "mouthbrooder" in n:
            return "Mouthbrooder"
        if "brood" in n or "reversed" in n or "seahorse" in n or "pipefish" in n:
            return "Male brooder"
        return "Livebearer (internal)"
    if fert == 2:
        return "Nest builder / guarder"
    # fert == 1
    if system >= 2:
        return "Harem / sex-changer"
    return "Pelagic pair spawn"

# expose a list of dicts
SPECIES = []
for sci, common, family, hab, fert, court, system, choice, note in _RAW:
    SPECIES.append(dict(
        scientific=sci, common=common, family=family, habitat=hab,
        c_fert=fert, c_court=court, c_system=system, c_choice=choice,
        mate_choice_index=_index(fert, court, system, choice), note=note,
        mating_mode=mating_mode(fert, court, system, note, family),
    ))

if __name__ == "__main__":
    import pandas as pd
    df = pd.DataFrame(SPECIES).sort_values("mate_choice_index")
    print(f"{len(df)} species, {df['family'].nunique()} families")
    print(df[["common","family","mate_choice_index"]].to_string(index=False))

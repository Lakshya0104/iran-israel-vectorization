"""
Iran-Israel Conflict Tweet Sentiment Dataset Generator
3000+ samples | 3 classes: Pro-Israel, Pro-Iran, Neutral/Ceasefire
Realistic tweet patterns: hashtags, @mentions, slang, mixed language
"""

import random, csv, json, re
random.seed(2024)

# ══════════════════════════════════════════════════════════════
#  VOCABULARY POOLS — realistic tweet language
# ══════════════════════════════════════════════════════════════

# ── Hashtags ───────────────────────────────────────────────────
TAGS_IL = [
    "#Israel","#IranAttack","#IsraelUnderAttack","#IDF","#StandWithIsrael",
    "#IsraelDefends","#IronDome","#Hamas","#Gaza","#TelAviv","#Jerusalem",
    "#IsraelStrong","#NeverAgain","#DefendIsrael","#IsraeliLives",
    "#Mossad","#Zionism","#IsraelWins","#MiddleEast","#IranIsMissileing",
]
TAGS_IR = [
    "#Iran","#Palestine","#FreePalestine","#Gaza","#IranStrikes",
    "#PalestineWillBeFree","#Resistance","#Hezbollah","#IRGC",
    "#IsraeliAggression","#GenocideInGaza","#ZionistRegime","#Tehran",
    "#IslamicRepublic","#AlAqsa","#IsraelWarCrimes","#BombedGaza",
    "#DeathToIsrael","#IranPower","#OperationTruePromise",
]
TAGS_NEU = [
    "#CeasefireNow","#MiddleEastCrisis","#WorldWar3","#StopTheWar",
    "#PeaceNotWar","#Civilians","#HumanitarianCrisis","#UNSecurity",
    "#NoMoreWar","#IranIsraelWar","#Diplomacy","#RefugeeCrisis",
    "#InnocentLives","#SaveCivilians","#WarCrimes","#HumanRights",
    "#GlobalPeace","#ConflictZone","#BreakingNews","#MiddleEastPeace",
]

# ── Accounts/mentions ──────────────────────────────────────────
ACCOUNTS_IL = ["@netanyahu","@IDF","@IsraelMFA","@IsraeliPM",
                "@EliezerCohen","@HerzogIsrael","@ILDefense"]
ACCOUNTS_IR = ["@PressTV","@khamenei_ir","@IRIranGov","@IRNAenglish",
                "@Iran_News_Today","@IRGC_Official"]
ACCOUNTS_NEU = ["@UN","@ICRC","@BBCWorld","@Reuters","@CNN",
                "@AlJazeera","@Guardian","@amnesty","@MSF"]

# ══════════════════════════════════════════════════════════════
#  TWEET TEMPLATES — 3 sentiment classes
# ══════════════════════════════════════════════════════════════

# ── CLASS 1: PRO-ISRAEL / ANTI-IRAN (label=2) ─────────────────
PRO_ISRAEL = [
    "Iran launched {n} missiles at Israel tonight and the world is silent. Absolutely unacceptable {tag1} {tag2}",
    "The IDF intercepted {pct}% of Iranian ballistic missiles. Iron Dome is a miracle of engineering {tag1}",
    "Every single country that cheers for Iran cheering on a terrorist state. Wake up {tag1} {tag2}",
    "Iran fired {n} drones at civilian areas. This is NOT a military operation, this is terrorism {tag1}",
    "Israel has the right to defend itself. Period. No conditions. No buts {tag1} {tag2}",
    "If any other country was hit by {n} missiles, the whole world would be screaming. Antisemitism is real {tag1}",
    "The Islamic Republic just confirmed it: they want to wipe Israel off the map. Never forget {tag1} {tag2}",
    "IDF response to Iran was PRECISE and PROPORTIONAL. That's what defense looks like {tag1}",
    "Iran's proxies — Hamas, Hezbollah, Houthis — all funded by Tehran. Connect the dots people {tag1} {tag2}",
    "Watching Iron Dome intercept missiles live on TV. The technology is insane. Praying for Israel {tag1}",
    "October 7 was the deadliest day for Jews since the Holocaust. And now Iran openly attacks. Enough {tag1} {tag2}",
    "Israel WARNS before striking. Iran fires {n} missiles at cities without warning. See the difference {tag1}",
    "The entire Arab world knew Iran was attacking. Not one of them tried to stop it {tag1} {tag2}",
    "Iron Dome just saved thousands of Israeli lives tonight. Hat tip to the engineers and pilots {tag1}",
    "Iran's supreme leader is a war criminal. The IRGC is a terrorist organisation {tag1} {tag2}",
    "Bibi is right. You don't negotiate with a regime that calls for your destruction {tag1}",
    "Israel is the only democracy in the Middle East and they're being attacked by a theocratic regime {tag1} {tag2}",
    "Every Iranian missile that targets Tel Aviv is a war crime under international law {tag1}",
    "The world stood with Ukraine. Why won't it stand with Israel now? Double standards {tag1} {tag2}",
    "{n} missiles fired at a country of 9 million people. This is an existential threat {tag1}",
    "IDF pilots are the most disciplined military force in the world. Proud of them {tag1} {tag2}",
    "Iran's nuclear program is the REAL threat to world peace. Not Israel {tag1}",
    "Israel didn't start this. Hamas started this. Iran funds Hamas. Simple math {tag1} {tag2}",
    "The Abraham Accords gave us hope. Iran is trying to destroy that hope with missiles {tag1}",
    "Air raid sirens in Tel Aviv again tonight. Israelis running to shelters at 2am. This is terror {tag1} {tag2}",
    "Jordan and Saudi helping intercept Iranian missiles. Arab nations protecting Israel. Historic {tag1}",
    "Iran attacking Israel directly is a massive escalation. The regime must face consequences {tag1} {tag2}",
    "Israeli hospitals treating patients in underground bunkers rn. Because of Iran {tag1}",
    "The IDF response was surgical. Iran's attack was indiscriminate. That's the difference {tag1} {tag2}",
    "Khamenei thinks he can threaten Israel forever with no cost. Tonight Israel sent a different message {tag1}",
]

# ── CLASS 0: PRO-IRAN / ANTI-ISRAEL (label=0) ─────────────────
PRO_IRAN = [
    "Israel has been bombing Gaza for months and NOW they're surprised Iran responded? {tag1} {tag2}",
    "Iran's strike was a direct response to the bombing of its consulate in Damascus. Self-defense {tag1}",
    "{n} Palestinians killed in Gaza. Where was the world then? But Iran responds and suddenly it's WW3 {tag1} {tag2}",
    "The Zionist regime killed Iranian diplomats on Syrian soil first. Iran had every right to respond {tag1}",
    "Iran only has {n} nuclear sites. Israel has hundreds of nukes and NEVER signed the NPT. Think about that {tag1} {tag2}",
    "The IRGC showed the world that Iran cannot be attacked without consequences {tag1}",
    "Israel assassinated an Iranian general and acted shocked when Iran retaliated. Cause and effect {tag1} {tag2}",
    "Operation True Promise wasn't random. It was a calculated, legal response to Israeli aggression {tag1}",
    "Why is Iran defending itself called terrorism but Israel bombing hospitals called defence? {tag1} {tag2}",
    "The Islamic Republic has shown more restraint than any country that's suffered Israeli attacks {tag1}",
    "Western media calling Iran the aggressor while ignoring {n} months of Gaza genocide {tag1} {tag2}",
    "Iran targeted military sites ONLY. Israel bombs refugee camps. Who's the terrorist here {tag1}",
    "Palestine has been occupied for 75 years. Iran is the only country doing something about it {tag1} {tag2}",
    "Hezbollah, Hamas, and Iran are not terrorists. They are a resistance movement {tag1}",
    "The axis of resistance is growing stronger. Zionism will fall {tag1} {tag2}",
    "Iran pre-warned 72 hours before striking. They followed international law more than Israel does {tag1}",
    "Every Iranian drone is a response to an Israeli bomb on Palestinian children {tag1} {tag2}",
    "Free Palestine means holding Israel accountable. Iran understands that {tag1}",
    "Death to apartheid. Death to colonialism. The resistance will not stop {tag1} {tag2}",
    "Israel annexes land illegally for decades and now acts like Iran is the aggressor {tag1}",
    "The IRGC successfully penetrated Israeli airspace. The Iron Dome has weaknesses {tag1} {tag2}",
    "If Israel can bomb Syria, Lebanon, Gaza simultaneously, Iran can defend itself {tag1}",
    "Khamenei: we will not start wars but we will FINISH them. Remember that Israel {tag1} {tag2}",
    "The resistance front is united: Iran, Hezbollah, Hamas, Houthis. Israel is surrounded {tag1}",
    "International law gives occupied peoples the right to resist. Full stop {tag1} {tag2}",
    "Iran's missiles were a message. Next time there will be no warning {tag1}",
    "The Muslim world is waking up. Israel's days of impunity are over {tag1} {tag2}",
    "Netanyahu bombed an Iranian embassy. An EMBASSY. And Iran is called the aggressor {tag1}",
    "Gaza has been under siege for {n} days. Iran broke that silence {tag1} {tag2}",
    "The Zionist project is built on stolen land. Resistance is legitimate {tag1}",
]

# ── CLASS 1: NEUTRAL / CEASEFIRE / HUMANITARIAN (label=1) ──────
NEUTRAL = [
    "Both sides claiming victory while civilians on all sides suffer. When does it end {tag1} {tag2}",
    "Regardless of who started it, {n} civilians caught in the crossfire deserve protection {tag1}",
    "This is not about religion. This is about geopolitics, oil, and power. Wake up {tag1} {tag2}",
    "My feed is split 50/50 between pro-Israel and pro-Iran. Everyone is certain they're right {tag1}",
    "The UN Security Council needs to convene immediately. Diplomacy is the only answer {tag1} {tag2}",
    "Iranian AND Israeli civilians are terrified tonight. Both deserve safety {tag1}",
    "We are literally watching the world edge toward a regional war in real time {tag1} {tag2}",
    "If this escalates further, oil prices will spike, global markets will crash. Not just missiles {tag1}",
    "Imagine being a regular Iranian just trying to live your life while your govt fires missiles {tag1} {tag2}",
    "Imagine being in Tel Aviv running to a shelter at midnight. No civilian deserves this {tag1}",
    "This conflict has been brewing for 45 years. There are no easy answers {tag1} {tag2}",
    "Every Twitter expert has suddenly become a Middle East foreign policy analyst tonight {tag1}",
    "{n} journalists have been killed covering this conflict. That alone should terrify you {tag1} {tag2}",
    "The arms dealers are the only winners in every war. Always follow the money {tag1}",
    "I genuinely don't know who is right anymore. All I know is people are dying {tag1} {tag2}",
    "Both Iran and Israel have legitimate security concerns. Dialogue is the only path {tag1}",
    "The humanitarian corridor must be opened. Food, water, medicine first. Politics later {tag1} {tag2}",
    "This generation will grow up knowing nothing but war if adults don't figure this out {tag1}",
    "American taxpayers funding weapons for both sides indirectly. Wild {tag1} {tag2}",
    "The Middle East has been a powder keg for 100 years. This is not new. This is history {tag1}",
    "Waiting to see if Iran retaliates to the retaliation of the retaliation. Absurd loop {tag1} {tag2}",
    "Red Cross calling for immediate access to conflict zones. Where is the global response {tag1}",
    "Children in Gaza don't care about geopolitics. They just want to sleep without bombs {tag1} {tag2}",
    "There's a reason oil just spiked {pct}%. Markets are pricing in WW3 right now {tag1}",
    "The people of Iran do not want this war. The regime does. Big difference {tag1} {tag2}",
    "The people of Israel do not want to die either. Government decisions have human costs {tag1}",
    "Ceasefire. Now. Both sides. No conditions. People are dying {tag1} {tag2}",
    "Praying for every civilian caught between governments they did not elect {tag1}",
    "{n} million people in the region now live under the threat of escalation. Unacceptable {tag1} {tag2}",
    "We have nuclear armed states posturing against each other. This is how WW3 actually starts {tag1}",
    "The only solution is a negotiated two-state framework. Everything else is just violence {tag1} {tag2}",
    "Both narratives exist. Both peoples have suffered. Dehumanising either side helps no one {tag1}",
    "Media from both sides is showing you what they want you to see. Think critically {tag1} {tag2}",
    "Hospitals in the region are overwhelmed. Doctors are heroes. Politicians are the problem {tag1}",
    "I just want this to stop. I don't care who wins the argument. I care who survives {tag1} {tag2}",
]

# ── Intensifiers + tweet-style variations ─────────────────────
PREFIXES = [
    "BREAKING: ","THREAD: ","Hot take: ","Unpopular opinion: ",
    "Just saw on the news: ","Can't sleep because ","Reminder that ",
    "Nobody is talking about how ","Real talk: ","This is insane. ",
    "Woke up to see ","My take: ","Hard truth: ","For the record: ",
    "RT if you agree: ","Am I the only one who thinks ","Honest question: ",
    "","","","","","","",  # empty = no prefix (most common)
]

SUFFIXES = [
    " RT this.","  Retweet if you agree."," This needs to be said.",
    " Share before it gets deleted."," Stay safe everyone.",
    " Praying for all civilians."," The world is watching.",
    " History will judge us."," Wake up world."," Thoughts?",
    " Am I wrong??"," Drop your thoughts below."," 🧵",
    "","","","","","","","","",  # empty most common
]

NUMBERS = [72, 120, 170, 200, 300, 350, 180, 250, 14000, 30000,
           33000, 7, 45, 186, 200, 1700, 500, 8, 12, 9, 100, 56]
PCTS    = [90, 95, 99, 85, 73, 12, 5, 3, 40, 15, 7, 25, 20, 8]

# ══════════════════════════════════════════════════════════════
#  GENERATE TWEETS
# ══════════════════════════════════════════════════════════════

CROSS_TAG_INJECTION = 0.18  # probability of injecting an opposite/ambiguous hashtag
LABEL_NOISE_RATE   = 0.12  # fraction of tweets with noisy/mislabeled sentiment

def pick_tags(tag_pool_primary, tag_pool_secondary=None, n=2):
    tags = []
    if random.random() < CROSS_TAG_INJECTION:
        other_pool = []
        if tag_pool_secondary is not None:
            other_pool = tag_pool_secondary
        else:
            other_pool = TAGS_IL + TAGS_IR + TAGS_NEU
        other_pool = [t for t in other_pool if t not in tag_pool_primary]
        if other_pool:
            tags.extend(random.sample(other_pool, 1))

    tags.extend(random.sample(tag_pool_primary, min(n, len(tag_pool_primary))))
    if tag_pool_secondary and random.random() > 0.5 and len(tags) < n:
        tags += random.sample(tag_pool_secondary, 1)

    return tags[:n]

def make_tweet(template, label):
    # Pick tag pools based on label
    if label == 2:
        primary, secondary = TAGS_IL, TAGS_NEU
    elif label == 0:
        primary, secondary = TAGS_IR, TAGS_NEU
    else:
        primary = TAGS_NEU
        secondary = random.choice([TAGS_IL, TAGS_IR])

    tags = pick_tags(primary, secondary)
    tag1 = tags[0] if len(tags) > 0 else ""
    tag2 = tags[1] if len(tags) > 1 else ""

    tweet = template.format(
        n=random.choice(NUMBERS),
        pct=random.choice(PCTS),
        tag1=tag1, tag2=tag2
    )

    # Add prefix/suffix
    prefix = random.choice(PREFIXES)
    suffix = random.choice(SUFFIXES)
    tweet = prefix + tweet + suffix

    # Occasional @mention injection
    if random.random() > 0.75:
        if label == 2:
            mention = random.choice(ACCOUNTS_IL + ACCOUNTS_NEU)
        elif label == 0:
            mention = random.choice(ACCOUNTS_IR + ACCOUNTS_NEU)
        else:
            mention = random.choice(ACCOUNTS_NEU)
        words = tweet.split()
        insert_pos = random.randint(1, max(1, len(words)//2))
        words.insert(insert_pos, mention)
        tweet = " ".join(words)

    # Clean up double spaces
    tweet = re.sub(r" +", " ", tweet).strip()
    return tweet

# ── Label map ─────────────────────────────────────────────────
LABEL_MAP = {
    0: ("pro_iran",    PRO_IRAN),
    1: ("neutral",     NEUTRAL),
    2: ("pro_israel",  PRO_ISRAEL),
}

# ── Generate: 1100 pro-israel, 1100 pro-iran, 900 neutral = 3100 ──
COUNTS = {2: 1100, 0: 1100, 1: 900}

rows = []
uid  = 1
for label, count in COUNTS.items():
    sentiment_name, templates = LABEL_MAP[label]
    # Cycle through templates to ensure variety
    for i in range(count):
        tmpl  = templates[i % len(templates)]
        tweet = make_tweet(tmpl, label)
        rows.append({
            "id":        uid,
            "text":      tweet,
            "label":     label,
            "sentiment": sentiment_name,
        })
        uid += 1

# Introduce a small amount of label noise to make the dataset more realistic
noise_count = int(len(rows) * LABEL_NOISE_RATE)
noisy_ids = random.sample(range(len(rows)), noise_count)
for idx in noisy_ids:
    original_label = rows[idx]["label"]
    alt_labels = [lbl for lbl in LABEL_MAP if lbl != original_label]
    new_label = random.choice(alt_labels)
    rows[idx]["label"] = new_label
    rows[idx]["sentiment"] = LABEL_MAP[new_label][0]

random.shuffle(rows)
for i, r in enumerate(rows):
    r["id"] = i + 1

# ── Stats ──────────────────────────────────────────────────────
print(f"Total tweets   : {len(rows)}")
print(f"Pro-Israel (2) : {sum(1 for r in rows if r['label']==2)}")
print(f"Neutral    (1) : {sum(1 for r in rows if r['label']==1)}")
print(f"Pro-Iran   (0) : {sum(1 for r in rows if r['label']==0)}")
print(f"Label noise    : {noise_count} tweets ({LABEL_NOISE_RATE*100:.0f}%)")
print(f"\nSample tweets:")
for r in rows[:6]:
    print(f"  [{r['sentiment']:<12}] {r['text'][:85]}...")

# ── Save ───────────────────────────────────────────────────────
with open("iran_israel_tweets.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["id","text","label","sentiment"])
    writer.writeheader()
    writer.writerows(rows)

with open("iran_israel_tweets.json", "w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2, ensure_ascii=False)

print(f"\nSaved: iran_israel_tweets.csv ({len(rows)} rows)")
print(f"Saved: iran_israel_tweets.json")

# Few shot prompts for different tasks

GSM8K_prompt = [
    {
        "role": "system",
        "content": "You are a prompt rewriter whose main goal is to rewrite the prompt given by the user in the most optimal way without losing any information in them."
    },
    {
        "role": "user",
        "content": (
            "I have a set of questions and/or statements, please REWRITE all the questions/statements so that they are in the most optimal order that is the easiest to understand. DO NOT ANSWER ANY OF THE QUESTIONS JUST REWRITE. Here are the instructions:\n"
            "how much will John pay in sales tax?\n"
            "the purchase amount is $120\n"
            "the sales tax rate is 8.25%."
        )
    },
    {
        "role": "assistant",
        "content": "What is the total sales tax John will pay on a $120 purchase at an 8.25% rate?"
    },
]

Code_prompt = [
    {
        "role": "system",
        "content": "You are a prompt rewriter whose main goal is to rewrite the prompt given by the user in the most optimal way without losing any information in them."
    },
    {
        "role": "user",
        "content": (
            "I have a set of questions and/or statements, please REWRITE all the questions/statements so that they are in the most optimal order that is the easiest to understand. DO NOT ANSWER ANY OF THE QUESTIONS JUST REWRITE. Here are the instructions:\n"
            "create a python function that takes two arguments"
            "check if both arguments are strings"
            "if not, raise TypeError"
            "return concatenated result"
        )
    },
    {
        "role": "assistant",
        "content": "Write a function that takes two arguments, checks if both are strings, raises a TypeError if not, and returns their concatenated result."
    },
]

DB_prompt = [
    {
        "role": "system",
        "content": "You are a prompt rewriter whose main goal is to rewrite the prompt given by the user in the most optimal way without losing any information in them."
    },
    {
        "role": "user",
        "content": (
            "I have a set of questions and/or statements, please REWRITE all the questions/statements so that they are in the most optimal order that is the easiest to understand. DO NOT ANSWER ANY OF THE QUESTIONS JUST REWRITE. Here are the instructions:\n"
            "Write SQL query to find active users"
            "active users are those who made at least 5 purchases"
            "each purchase is in the purchases table"
            "users are in the users table"
            "return user names only"
        )
    },
    {
        "role": "assistant",
        "content": "Write an SQL query to find the names of users who have made at least 5 purchases, using the users and purchases tables."
    },
]

D2T_prompt = [
    {
        "role": "system",
        "content": "You are a prompt rewriter whose main goal is to rewrite the prompt given by the user in the most optimal way without losing any information in them."
    },
    {
        "role": "user",
        "content": (
            "I have a set of questions and/or statements, please REWRITE all the questions/statements so that they are in the most optimal order that is the easiest to understand. DO NOT ANSWER ANY OF THE QUESTIONS JUST REWRITE. Here are the instructions:\n"
"""User Instruction 1: I am providing you a table. You must produce a short one-sentence description of the table. The description should be at most 30 words. In followup turns, I will provide further information about the table. You must update your description to take information from all the turns into account. At each turn, only respond with a single sentence describing the table. Do not respond with any other text.

Table:
<table>
<tr> <th colspan=1 rowspan=1 > Pos </th><th colspan=1 rowspan=1 > No. </th><th colspan=1 rowspan=1 > Driver </th><th colspan=1 rowspan=1 > Team </th><th colspan=1 rowspan=1 > Manufacturer </th><th colspan=1 rowspan=1 > Time </th><th colspan=1 rowspan=1 > Speed </th></tr>
<tr> <th colspan=1 rowspan=1 > 1 </th><td colspan=1 rowspan=1 > 41 </td><td colspan=1 rowspan=1 > Kurt Busch </td><td colspan=1 rowspan=1 > Stewart-Haas Racing </td><td colspan=1 rowspan=1 > Chevrolet </td><td colspan=1 rowspan=1 > 25.928 </td><td colspan=1 rowspan=1 > 138.846 </td></tr>
<tr> <th colspan=1 rowspan=1 > 2 </th><td colspan=1 rowspan=1 > 11 </td><td colspan=1 rowspan=1 > Denny Hamlin </td><td colspan=1 rowspan=1 > Joe Gibbs Racing </td><td colspan=1 rowspan=1 > Toyota </td><td colspan=1 rowspan=1 > 25.999 </td><td colspan=1 rowspan=1 > 138.467 </td></tr>
</table>

User Instruction 2: Here's 2 examples of descriptions of other tables. You can use it as references to understand how to describe the table:
Hagino's fellow countryman Daiya Seto captured the bronze in 4:09.71, to give Japan two swimmers on the same Olympic podium.
In 2017, Stamenković appeared in the Kazakhstan Premier League for Irtysh Pavlodar.
User Instruction 3: 'I am now giving you a version of the table with some cells highlights (by adding a highlighted class to each cell). You can now update your description of the table, which must now focus on the highlighted cells in the table.

<table>
<tr> <th colspan=1 rowspan=1 > Pos </th><th colspan=1 rowspan=1 > No. </th><th colspan=1 rowspan=1 > Driver </th><th colspan=1 rowspan=1 > Team </th><th colspan=1 rowspan=1 > Manufacturer </th><th colspan=1 rowspan=1 > Time </th><th colspan=1 rowspan=1 > Speed </th></tr>
<tr> <th class="highlighted" colspan=1 rowspan=1 > 1 </th><td colspan=1 rowspan=1 > 41 </td><td class="highlighted" colspan=1 rowspan=1 > Kurt Busch </td><td colspan=1 rowspan=1 > Stewart-Haas Racing </td><td colspan=1 rowspan=1 > Chevrolet </td><td colspan=1 rowspan=1 > 25.928 </td><td colspan=1 rowspan=1 > 138.846 </td></tr>
<tr> <th class="highlighted" colspan=1 rowspan=1 > 2 </th><td colspan=1 rowspan=1 > 11 </td><td class="highlighted" colspan=1 rowspan=1 > Denny Hamlin </td><td colspan=1 rowspan=1 > Joe Gibbs Racing </td><td colspan=1 rowspan=1 > Toyota </td><td colspan=1 rowspan=1 > 25.999 </td><td colspan=1 rowspan=1 > 138.467 </td></tr>
</table>

User Instruction 4: Oh also, Here's additional metadata about the table. The table_webpage_url is http://en.wikipedia.org/wiki/2016_Good_Sam_500.
User Instruction 5: FYI, Here's additional metadata about the table. The table_page_title is 2016 Good Sam 500."""
        )
    },
    {
        "role": "assistant",
        "content": """Provide a short one-sentence description of the following table. Only focus on the highlighted cells. In future turns, update the description with any new provided information. Output only one sentence per turn, no other text.

Table:
<table>
<tr> <th colspan=1 rowspan=1 > Pos </th><th colspan=1 rowspan=1 > No. </th><th colspan=1 rowspan=1 > Driver </th><th colspan=1 rowspan=1 > Team </th><th colspan=1 rowspan=1 > Manufacturer </th><th colspan=1 rowspan=1 > Time </th><th colspan=1 rowspan=1 > Speed </th></tr>
<tr> <th class="highlighted" colspan=1 rowspan=1 > 1 </th><td colspan=1 rowspan=1 > 41 </td><td class="highlighted" colspan=1 rowspan=1 > Kurt Busch </td><td colspan=1 rowspan=1 > Stewart-Haas Racing </td><td colspan=1 rowspan=1 > Chevrolet </td><td colspan=1 rowspan=1 > 25.928 </td><td colspan=1 rowspan=1 > 138.846 </td></tr>
<tr> <th class="highlighted" colspan=1 rowspan=1 > 2 </th><td colspan=1 rowspan=1 > 11 </td><td class="highlighted" colspan=1 rowspan=1 > Denny Hamlin </td><td colspan=1 rowspan=1 > Joe Gibbs Racing </td><td colspan=1 rowspan=1 > Toyota </td><td colspan=1 rowspan=1 > 25.999 </td><td colspan=1 rowspan=1 > 138.467 </td></tr>
</table>

Here is additional metadata about the table:
The table_webpage_url is http://en.wikipedia.org/wiki/2016_Good_Sam_500.
The table_page_title is 2016 Good Sam 500.

Here are 4 example descriptions for reference:
Hagino's fellow countryman Daiya Seto captured the bronze in 4:09.71, to give Japan two swimmers on the same Olympic podium.
In 2017, Stamenković appeared in the Kazakhstan Premier League for Irtysh Pavlodar."""
    },
]

Summary_prompt = [
    {
        "role": "system",
        "content": "You are a prompt rewriter whose main goal is to rewrite the prompt given by the user in the most optimal way without losing any information in them."
    },
    {
        "role": "user",
        "content": (
"""I have a set of questions and/or statements, please REWRITE all the questions/statements so that they are in the most optimal order that is the easiest to understand. DO NOT ANSWER ANY OF THE QUESTIONS JUST REWRITE. Here are the instructions:\n
User Instruction 1: I need a summary of 8 documents, on query: What are the names of the people, cities, and organizations who are involved in construction efforts? I’ll give the docs as I get them, consider all of them.
Docs 1-2: Munich 1856: Four maps that will change your view of the city\nA mental asylum, where today young people are said to meet.\nA crypt chapel, where they are now digging tunnels for the S-Bahn.\nAllotment holders cultivate the soil of former farmers.\nThe oldest official map of Munich brings captivating stories to light.\nIt is annoying when geographical maps are not up-to-date.\nAnyone who has ever got worked up because the car's sat-nav is showing a green field instead of a bypass knows that.\nThe historical maps of the digital BayernAtlas, an offering from the State Government's Geoportal Bayern, are anything but up-to-date \u2013 and yet it is precisely for this reason that they are so informative.\nEspecially when one compares them with current online maps.\nThen it becomes clear how the towns and municipalities in the distribution area of Munich's Merkur newspaper have changed since the 19th century.
Pasture fence project is fundamental\nThe Fischerbach pasture fence project is a successful project and will be continued next year.\nMembers of the Bundestag for the CDU Peter Wei\u00df and Kordula Kovac got a first-hand view.\nRepresentatives from the Fischerbach local CDU group met with mayor Thomas Schneider and the two representatives at Rechtgrabenhof farm, belonging to the Schwendemann family.\nTo start with, Markus Schwendemann gave insights into his business, in which he cultivates 55 hectares of pasture and 40 hectares of woodland, keeps 40 milk cows and just as many young stock, and accommodates five apartments of holiday makers.\nSchwendemann says that, for him, there is no alternative to the \"Black Forest milk\" marketing channel he uses for the milk from his cows.\nOnly regional produce is served to the holiday-makers.\nSince he is also BLHV chairman for Fischerbach, he described the pasture fence project as \"fundamentally important\".\nI can't make a profit on that kind of steep-sloping ground.\nGrassland areas make for a lot of work, a lot has to be done with the string trimmer.

User Instruction 2: Just got two more.
Docs 3-4: Tumlingen is celebrating 750 years of the St. Hilarius Church\n750 years of the St. Hilarius Church: To mark the anniversary of the mountain church, invitations are being issued to a serenade in the old cemetery grounds in Tumlingen on Sunday 16th July at 4.30pm.\nDue to its idyllic setting and beautiful view over the village, the historical site of St. Hilarius is a beloved place to tarry a while.\nVisitors to the sixth benefit serenade can let themselves be enchanted by this powerful place.\nThe Friends and Supporters of Tumlingen have made the rescue and preservation of the oldest building in the area their concern.\nWalter Martini stresses: \"It is a one-of-a-kind rural jewel\".\nThe St. Hilarius chapel was first mentioned in records in 1267 in the W\u00fcrttemberg register: Count Hermann von Sulz sold \"the mountain with the parish church for Tumlingen on it, together with the Bailiwick\" to Master Eberhard von Horb and his brother Dietrich for 20 silver marks.\nAccording to the text of the official document, aside from the former parish church, a lordly farm also stood on the Tumlingen mountain, which itself owned the church, or the right of patronage, and not the other way around.\nPeople therefore also lived there too.\nAs is apparent from the upper administrative office for Freudenstadt's description in 1858, the \"mountain\" was, alongside the lower village and the upper village that can just about be made out still in the appearance of Tumlingen today, the village's third, grand centre of settlement.
Association wants to have wind turbine approval examined\n\"Boisheim wehrt sich\", sees no legal basis for the construction and has therefore turned to the District Administrator.\nThe association \"Boisheim wehrt sich\" has called on District Administrator Andreas Coenen (CDU) to retract his approval for the construction of planned wind turbines in Boisheimer Nette.\nThey say it was unlawful.\nThe association argues that at the time the approval was given on 30th December 2016, no wind turbines could have been approved in Bosheimer Nette, since at that time only one wind energy concentration zone existed in Viersen, on the city boundary with Schwalmtal.\nConstruction and operation of wind energy plants are in principle possible within this concentration zone, but at the same time, the concentration zone has an exclusion against the use of wind energy for the rest of the city area.\nSuch zones are said to prevent \"asparagusisation\" of the landscape.\nThe city of Viersen had in fact passed a resolution for a second wind energy concentration zone in Boisheimer Nette a few days earlier, through an amendment to the local land-use plan \u2013 but the D\u00fcsseldorf regional government had not approved it due to a formal error.\n\"Due to the lack of legal planning requirements, the approval decision that was issued is still unlawful, and should therefore be retracted as well,\" said Christoph Erkens, chairman of \"Boisheim wehrt sich\".\nThe current state of affairs: the city council once again passed a resolution for the wind concentration zone in Boisheimer Nette before the summer break; approval from the regional government is still pending.

User Instruction 3: Here’s a new batch.
Docs 5-7: Town and dealers plan \"hometown shopping\" weekend\nBad Langensalzaer and people from the region are to newly discover the shopping opportunities in their town centre in September.\nThe trade association and town authorities want to use the weekend of the 8th and 9th September to entice residents to a shopping spree in the town centre.\nAs chairman of the trade association Christian K\u00f6nig announced, the \"home-town shopping\" weekend is meant to make customers aware of what offers their own town has available aside from the usual supermarket goods.\nA market with local suppliers is thus planned, similar to the weekly market.\nIn addition, homegrown companies are to give showcases, with their apprenticeship places.\nFurthermore, the businesses in the town centre are to invite people to come \"midnight shopping\".\nThe campaign is not only for residents of the town and the nearby region.\nAdverts on the radio, an online video and printed bags that allude to the campaign weekend are in preparation, Christian K\u00f6nig tells us.\nTown authorities and the trade association are cooperating with the Chamber of Industry and Commerce.
Max Menge works as volunteer\nDuring your voluntary service year, you're working at many locations simultaneously.\nWhat are all the different places?\nMenge: Normally in Steinfurt in the furniture exchange, and once a week I help out here in Greven in the M\u00f6belbr\u00fccke.\nAnd I have also worked at the full-day school in their supplementary activities.\nWhat are your experiences of that?\nMenge: It is a nice, broad spectrum that I see here.\nI deal with different people: with adolescents, with refugees, with young adults.\nYou are 26 years old now, so a volunteer with a late-calling.\nHow did that come about?
Municipal council wants common room\nIn the end, the decision was clear: with two dissenting votes, the municipal council of Fluorn-Winzeln passed a resolution for a common room in the future retirement home.\nAhead of the debate, there is an important piece of information to be had: the purchase contract for the as yet still needed plot section has been signed, reports mayor Bernhard Tjaden.\nAdmittedly it will still be a while until the plot, which up until now has belonged to the church parish, comes into the possession of the municipality, but nevertheless, there are now no longer any planning obstacles for the project that has to be materialised by a private investor.\nThat is why there is a debate concerning a common room.\nOne of the two investors that are still on board has offered to accommodate the municipality.\nFor a special price, it would provide a living space \u2013 of around 55 square metres \u2013 and an adjacent terrace with no less than 70 square metres on the ground floor.\nWhat the management and the majority of the municipal council have in mind is not simply to set aside such a common room as a meeting place only for the residents of the senior citizens' accommodation, but for it to be open to the community.\nTalk of a \"knitting caf\u00e9\" was making the rounds, for example.\nThe decision, which was already pre-determined last Autumn through the allocation of resources for the current budget, is sweetened by funding opportunities.

User Instruction 4: I've got one more.
Doc 8: Home connections prove to be very complex\nThe building works in Villinger Stra\u00dfe are in full throw and \"are more or less on schedule,\" according to the manager of the construction yard, Fred Heinze.\nIn terms of home connections, the situation is more complicated than expected, since lots of things are hidden in the substrate that were not noted in the plans.\nThe traffic, which always disrupts building works, poses a big problem for the Stumpp company construction crew.\n\"The ignorance of car drivers around red lights and barriers is really huge,\" reports Heinze.\nThe street is closed off and only the public busses are allowed to drive through.\nAngelo Caruso, Marvin Kostorz and Tim Kostorz from the Stumpp company have been engaged in the works on the bridges for weeks.\n\"The hot days were pretty exhausting, for sure.\nBut luckily we were supplied with cold mineral water by our friend Fritz Matt,\" explained Marvin Kostorz.\n
"""
        )
    },
    {
        "role": "assistant",
        "content": 
"""Write a summary about the following 8 documents, on the query: What are the names of the people, cities, and organizations who are involved in construction efforts?

Documents:
Munich 1856: Four maps that will change your view of the city\nA mental asylum, where today young people are said to meet.\nA crypt chapel, where they are now digging tunnels for the S-Bahn.\nAllotment holders cultivate the soil of former farmers.\nThe oldest official map of Munich brings captivating stories to light.\nIt is annoying when geographical maps are not up-to-date.\nAnyone who has ever got worked up because the car's sat-nav is showing a green field instead of a bypass knows that.\nThe historical maps of the digital BayernAtlas, an offering from the State Government's Geoportal Bayern, are anything but up-to-date \u2013 and yet it is precisely for this reason that they are so informative.\nEspecially when one compares them with current online maps.\nThen it becomes clear how the towns and municipalities in the distribution area of Munich's Merkur newspaper have changed since the 19th century.
Pasture fence project is fundamental\nThe Fischerbach pasture fence project is a successful project and will be continued next year.\nMembers of the Bundestag for the CDU Peter Wei\u00df and Kordula Kovac got a first-hand view.\nRepresentatives from the Fischerbach local CDU group met with mayor Thomas Schneider and the two representatives at Rechtgrabenhof farm, belonging to the Schwendemann family.\nTo start with, Markus Schwendemann gave insights into his business, in which he cultivates 55 hectares of pasture and 40 hectares of woodland, keeps 40 milk cows and just as many young stock, and accommodates five apartments of holiday makers.\nSchwendemann says that, for him, there is no alternative to the \"Black Forest milk\" marketing channel he uses for the milk from his cows.\nOnly regional produce is served to the holiday-makers.\nSince he is also BLHV chairman for Fischerbach, he described the pasture fence project as \"fundamentally important\".\nI can't make a profit on that kind of steep-sloping ground.\nGrassland areas make for a lot of work, a lot has to be done with the string trimmer.
Tumlingen is celebrating 750 years of the St. Hilarius Church\n750 years of the St. Hilarius Church: To mark the anniversary of the mountain church, invitations are being issued to a serenade in the old cemetery grounds in Tumlingen on Sunday 16th July at 4.30pm.\nDue to its idyllic setting and beautiful view over the village, the historical site of St. Hilarius is a beloved place to tarry a while.\nVisitors to the sixth benefit serenade can let themselves be enchanted by this powerful place.\nThe Friends and Supporters of Tumlingen have made the rescue and preservation of the oldest building in the area their concern.\nWalter Martini stresses: \"It is a one-of-a-kind rural jewel\".\nThe St. Hilarius chapel was first mentioned in records in 1267 in the W\u00fcrttemberg register: Count Hermann von Sulz sold \"the mountain with the parish church for Tumlingen on it, together with the Bailiwick\" to Master Eberhard von Horb and his brother Dietrich for 20 silver marks.\nAccording to the text of the official document, aside from the former parish church, a lordly farm also stood on the Tumlingen mountain, which itself owned the church, or the right of patronage, and not the other way around.\nPeople therefore also lived there too.\nAs is apparent from the upper administrative office for Freudenstadt's description in 1858, the \"mountain\" was, alongside the lower village and the upper village that can just about be made out still in the appearance of Tumlingen today, the village's third, grand centre of settlement.
Association wants to have wind turbine approval examined\n\"Boisheim wehrt sich\", sees no legal basis for the construction and has therefore turned to the District Administrator.\nThe association \"Boisheim wehrt sich\" has called on District Administrator Andreas Coenen (CDU) to retract his approval for the construction of planned wind turbines in Boisheimer Nette.\nThey say it was unlawful.\nThe association argues that at the time the approval was given on 30th December 2016, no wind turbines could have been approved in Bosheimer Nette, since at that time only one wind energy concentration zone existed in Viersen, on the city boundary with Schwalmtal.\nConstruction and operation of wind energy plants are in principle possible within this concentration zone, but at the same time, the concentration zone has an exclusion against the use of wind energy for the rest of the city area.\nSuch zones are said to prevent \"asparagusisation\" of the landscape.\nThe city of Viersen had in fact passed a resolution for a second wind energy concentration zone in Boisheimer Nette a few days earlier, through an amendment to the local land-use plan \u2013 but the D\u00fcsseldorf regional government had not approved it due to a formal error.\n\"Due to the lack of legal planning requirements, the approval decision that was issued is still unlawful, and should therefore be retracted as well,\" said Christoph Erkens, chairman of \"Boisheim wehrt sich\".\nThe current state of affairs: the city council once again passed a resolution for the wind concentration zone in Boisheimer Nette before the summer break; approval from the regional government is still pending.
Town and dealers plan \"hometown shopping\" weekend\nBad Langensalzaer and people from the region are to newly discover the shopping opportunities in their town centre in September.\nThe trade association and town authorities want to use the weekend of the 8th and 9th September to entice residents to a shopping spree in the town centre.\nAs chairman of the trade association Christian K\u00f6nig announced, the \"home-town shopping\" weekend is meant to make customers aware of what offers their own town has available aside from the usual supermarket goods.\nA market with local suppliers is thus planned, similar to the weekly market.\nIn addition, homegrown companies are to give showcases, with their apprenticeship places.\nFurthermore, the businesses in the town centre are to invite people to come \"midnight shopping\".\nThe campaign is not only for residents of the town and the nearby region.\nAdverts on the radio, an online video and printed bags that allude to the campaign weekend are in preparation, Christian K\u00f6nig tells us.\nTown authorities and the trade association are cooperating with the Chamber of Industry and Commerce.
Max Menge works as volunteer\nDuring your voluntary service year, you're working at many locations simultaneously.\nWhat are all the different places?\nMenge: Normally in Steinfurt in the furniture exchange, and once a week I help out here in Greven in the M\u00f6belbr\u00fccke.\nAnd I have also worked at the full-day school in their supplementary activities.\nWhat are your experiences of that?\nMenge: It is a nice, broad spectrum that I see here.\nI deal with different people: with adolescents, with refugees, with young adults.\nYou are 26 years old now, so a volunteer with a late-calling.\nHow did that come about?
Municipal council wants common room\nIn the end, the decision was clear: with two dissenting votes, the municipal council of Fluorn-Winzeln passed a resolution for a common room in the future retirement home.\nAhead of the debate, there is an important piece of information to be had: the purchase contract for the as yet still needed plot section has been signed, reports mayor Bernhard Tjaden.\nAdmittedly it will still be a while until the plot, which up until now has belonged to the church parish, comes into the possession of the municipality, but nevertheless, there are now no longer any planning obstacles for the project that has to be materialised by a private investor.\nThat is why there is a debate concerning a common room.\nOne of the two investors that are still on board has offered to accommodate the municipality.\nFor a special price, it would provide a living space \u2013 of around 55 square metres \u2013 and an adjacent terrace with no less than 70 square metres on the ground floor.\nWhat the management and the majority of the municipal council have in mind is not simply to set aside such a common room as a meeting place only for the residents of the senior citizens' accommodation, but for it to be open to the community.\nTalk of a \"knitting caf\u00e9\" was making the rounds, for example.\nThe decision, which was already pre-determined last Autumn through the allocation of resources for the current budget, is sweetened by funding opportunities.
Home connections prove to be very complex\nThe building works in Villinger Stra\u00dfe are in full throw and \"are more or less on schedule,\" according to the manager of the construction yard, Fred Heinze.\nIn terms of home connections, the situation is more complicated than expected, since lots of things are hidden in the substrate that were not noted in the plans.\nThe traffic, which always disrupts building works, poses a big problem for the Stumpp company construction crew.\n\"The ignorance of car drivers around red lights and barriers is really huge,\" reports Heinze.\nThe street is closed off and only the public busses are allowed to drive through.\nAngelo Caruso, Marvin Kostorz and Tim Kostorz from the Stumpp company have been engaged in the works on the bridges for weeks.\n\"The hot days were pretty exhausting, for sure.\nBut luckily we were supplied with cold mineral water by our friend Fritz Matt,\" explained Marvin Kostorz.\n
"""
    },
]

# for Claude, which requires a separate system prompt

GSM8K_prompt_claude = [
    {
        "role": "user",
        "content": (
            "I have a set of questions and/or statements, please REWRITE all the questions/statements so that they are in the most optimal order that is the easiest to understand. DO NOT ANSWER ANY OF THE QUESTIONS JUST REWRITE. Here are the instructions:\n"
            "how much will John pay in sales tax?\n"
            "the purchase amount is $120\n"
            "the sales tax rate is 8.25%."
        )
    },
    {
        "role": "assistant",
        "content": "What is the total sales tax John will pay on a $120 purchase at an 8.25% rate?"
    },
]

GSM8K_system_claude = "You are a prompt rewriter whose main goal is to rewrite the prompt given by the user in the most optimal way without losing any information in them."

Code_prompt_claude = [
    {
        "role": "user",
        "content": (
            "I have a set of questions and/or statements, please REWRITE all the questions/statements so that they are in the most optimal order that is the easiest to understand. DO NOT ANSWER ANY OF THE QUESTIONS JUST REWRITE. Here are the instructions:\n"
            "create a python function that takes two arguments"
            "check if both arguments are strings"
            "if not, raise TypeError"
            "return concatenated result"
        )
    },
    {
        "role": "assistant",
        "content": "Write a function that takes two arguments, checks if both are strings, raises a TypeError if not, and returns their concatenated result."
    },
]

Code_system_claude = "You are a prompt rewriter whose main goal is to rewrite the prompt given by the user in the most optimal way without losing any information in them."

DB_prompt_claude = [
    {
        "role": "user",
        "content": (
            "I have a set of questions and/or statements, please REWRITE all the questions/statements so that they are in the most optimal order that is the easiest to understand. DO NOT ANSWER ANY OF THE QUESTIONS JUST REWRITE. Here are the instructions:\n"
            "Write SQL query to find active users"
            "active users are those who made at least 5 purchases"
            "each purchase is in the purchases table"
            "users are in the users table"
            "return user names only"
        )
    },
    {
        "role": "assistant",
        "content": "Write an SQL query to find the names of users who have made at least 5 purchases, using the users and purchases tables."
    },
]

DB_system_claude = "You are a prompt rewriter whose main goal is to rewrite the prompt given by the user in the most optimal way without losing any information in them."

D2T_prompt_claude = [
    {
        "role": "user",
        "content": (
            "I have a set of questions and/or statements, please REWRITE all the questions/statements so that they are in the most optimal order that is the easiest to understand. DO NOT ANSWER ANY OF THE QUESTIONS JUST REWRITE. Here are the instructions:\n"
"""User Instruction 1: I am providing you a table. You must produce a short one-sentence description of the table. The description should be at most 30 words. In followup turns, I will provide further information about the table. You must update your description to take information from all the turns into account. At each turn, only respond with a single sentence describing the table. Do not respond with any other text.

Table:
<table>
<tr> <th colspan=1 rowspan=1 > Pos </th><th colspan=1 rowspan=1 > No. </th><th colspan=1 rowspan=1 > Driver </th><th colspan=1 rowspan=1 > Team </th><th colspan=1 rowspan=1 > Manufacturer </th><th colspan=1 rowspan=1 > Time </th><th colspan=1 rowspan=1 > Speed </th></tr>
<tr> <th colspan=1 rowspan=1 > 1 </th><td colspan=1 rowspan=1 > 41 </td><td colspan=1 rowspan=1 > Kurt Busch </td><td colspan=1 rowspan=1 > Stewart-Haas Racing </td><td colspan=1 rowspan=1 > Chevrolet </td><td colspan=1 rowspan=1 > 25.928 </td><td colspan=1 rowspan=1 > 138.846 </td></tr>
<tr> <th colspan=1 rowspan=1 > 2 </th><td colspan=1 rowspan=1 > 11 </td><td colspan=1 rowspan=1 > Denny Hamlin </td><td colspan=1 rowspan=1 > Joe Gibbs Racing </td><td colspan=1 rowspan=1 > Toyota </td><td colspan=1 rowspan=1 > 25.999 </td><td colspan=1 rowspan=1 > 138.467 </td></tr>
</table>

User Instruction 2: Here's 2 examples of descriptions of other tables. You can use it as references to understand how to describe the table:
Hagino's fellow countryman Daiya Seto captured the bronze in 4:09.71, to give Japan two swimmers on the same Olympic podium.
In 2017, Stamenković appeared in the Kazakhstan Premier League for Irtysh Pavlodar.
User Instruction 3: 'I am now giving you a version of the table with some cells highlights (by adding a highlighted class to each cell). You can now update your description of the table, which must now focus on the highlighted cells in the table.

<table>
<tr> <th colspan=1 rowspan=1 > Pos </th><th colspan=1 rowspan=1 > No. </th><th colspan=1 rowspan=1 > Driver </th><th colspan=1 rowspan=1 > Team </th><th colspan=1 rowspan=1 > Manufacturer </th><th colspan=1 rowspan=1 > Time </th><th colspan=1 rowspan=1 > Speed </th></tr>
<tr> <th class="highlighted" colspan=1 rowspan=1 > 1 </th><td colspan=1 rowspan=1 > 41 </td><td class="highlighted" colspan=1 rowspan=1 > Kurt Busch </td><td colspan=1 rowspan=1 > Stewart-Haas Racing </td><td colspan=1 rowspan=1 > Chevrolet </td><td colspan=1 rowspan=1 > 25.928 </td><td colspan=1 rowspan=1 > 138.846 </td></tr>
<tr> <th class="highlighted" colspan=1 rowspan=1 > 2 </th><td colspan=1 rowspan=1 > 11 </td><td class="highlighted" colspan=1 rowspan=1 > Denny Hamlin </td><td colspan=1 rowspan=1 > Joe Gibbs Racing </td><td colspan=1 rowspan=1 > Toyota </td><td colspan=1 rowspan=1 > 25.999 </td><td colspan=1 rowspan=1 > 138.467 </td></tr>
</table>

User Instruction 4: Oh also, Here's additional metadata about the table. The table_webpage_url is http://en.wikipedia.org/wiki/2016_Good_Sam_500.
User Instruction 5: FYI, Here's additional metadata about the table. The table_page_title is 2016 Good Sam 500."""
        )
    },
    {
        "role": "assistant",
        "content": """Provide a short one-sentence description of the following table. Only focus on the highlighted cells. In future turns, update the description with any new provided information. Output only one sentence per turn, no other text.

Table:
<table>
<tr> <th colspan=1 rowspan=1 > Pos </th><th colspan=1 rowspan=1 > No. </th><th colspan=1 rowspan=1 > Driver </th><th colspan=1 rowspan=1 > Team </th><th colspan=1 rowspan=1 > Manufacturer </th><th colspan=1 rowspan=1 > Time </th><th colspan=1 rowspan=1 > Speed </th></tr>
<tr> <th class="highlighted" colspan=1 rowspan=1 > 1 </th><td colspan=1 rowspan=1 > 41 </td><td class="highlighted" colspan=1 rowspan=1 > Kurt Busch </td><td colspan=1 rowspan=1 > Stewart-Haas Racing </td><td colspan=1 rowspan=1 > Chevrolet </td><td colspan=1 rowspan=1 > 25.928 </td><td colspan=1 rowspan=1 > 138.846 </td></tr>
<tr> <th class="highlighted" colspan=1 rowspan=1 > 2 </th><td colspan=1 rowspan=1 > 11 </td><td class="highlighted" colspan=1 rowspan=1 > Denny Hamlin </td><td colspan=1 rowspan=1 > Joe Gibbs Racing </td><td colspan=1 rowspan=1 > Toyota </td><td colspan=1 rowspan=1 > 25.999 </td><td colspan=1 rowspan=1 > 138.467 </td></tr>
</table>

Here is additional metadata about the table:
The table_webpage_url is http://en.wikipedia.org/wiki/2016_Good_Sam_500.
The table_page_title is 2016 Good Sam 500.

Here are 4 example descriptions for reference:
Hagino's fellow countryman Daiya Seto captured the bronze in 4:09.71, to give Japan two swimmers on the same Olympic podium.
In 2017, Stamenković appeared in the Kazakhstan Premier League for Irtysh Pavlodar."""
    },
]

D2T_system_claude = "You are a prompt rewriter whose main goal is to rewrite the prompt given by the user in the most optimal way without losing any information in them."

Summary_prompt_claude = [
    {
        "role": "user",
        "content": (
"""I have a set of questions and/or statements, please REWRITE all the questions/statements so that they are in the most optimal order that is the easiest to understand. DO NOT ANSWER ANY OF THE QUESTIONS JUST REWRITE. Here are the instructions:\n
User Instruction 1: I need a summary of 8 documents, on query: What are the names of the people, cities, and organizations who are involved in construction efforts? I’ll give the docs as I get them, consider all of them.
Docs 1-2: Munich 1856: Four maps that will change your view of the city\nA mental asylum, where today young people are said to meet.\nA crypt chapel, where they are now digging tunnels for the S-Bahn.\nAllotment holders cultivate the soil of former farmers.\nThe oldest official map of Munich brings captivating stories to light.\nIt is annoying when geographical maps are not up-to-date.\nAnyone who has ever got worked up because the car's sat-nav is showing a green field instead of a bypass knows that.\nThe historical maps of the digital BayernAtlas, an offering from the State Government's Geoportal Bayern, are anything but up-to-date \u2013 and yet it is precisely for this reason that they are so informative.\nEspecially when one compares them with current online maps.\nThen it becomes clear how the towns and municipalities in the distribution area of Munich's Merkur newspaper have changed since the 19th century.
Pasture fence project is fundamental\nThe Fischerbach pasture fence project is a successful project and will be continued next year.\nMembers of the Bundestag for the CDU Peter Wei\u00df and Kordula Kovac got a first-hand view.\nRepresentatives from the Fischerbach local CDU group met with mayor Thomas Schneider and the two representatives at Rechtgrabenhof farm, belonging to the Schwendemann family.\nTo start with, Markus Schwendemann gave insights into his business, in which he cultivates 55 hectares of pasture and 40 hectares of woodland, keeps 40 milk cows and just as many young stock, and accommodates five apartments of holiday makers.\nSchwendemann says that, for him, there is no alternative to the \"Black Forest milk\" marketing channel he uses for the milk from his cows.\nOnly regional produce is served to the holiday-makers.\nSince he is also BLHV chairman for Fischerbach, he described the pasture fence project as \"fundamentally important\".\nI can't make a profit on that kind of steep-sloping ground.\nGrassland areas make for a lot of work, a lot has to be done with the string trimmer.

User Instruction 2: Just got two more.
Docs 3-4: Tumlingen is celebrating 750 years of the St. Hilarius Church\n750 years of the St. Hilarius Church: To mark the anniversary of the mountain church, invitations are being issued to a serenade in the old cemetery grounds in Tumlingen on Sunday 16th July at 4.30pm.\nDue to its idyllic setting and beautiful view over the village, the historical site of St. Hilarius is a beloved place to tarry a while.\nVisitors to the sixth benefit serenade can let themselves be enchanted by this powerful place.\nThe Friends and Supporters of Tumlingen have made the rescue and preservation of the oldest building in the area their concern.\nWalter Martini stresses: \"It is a one-of-a-kind rural jewel\".\nThe St. Hilarius chapel was first mentioned in records in 1267 in the W\u00fcrttemberg register: Count Hermann von Sulz sold \"the mountain with the parish church for Tumlingen on it, together with the Bailiwick\" to Master Eberhard von Horb and his brother Dietrich for 20 silver marks.\nAccording to the text of the official document, aside from the former parish church, a lordly farm also stood on the Tumlingen mountain, which itself owned the church, or the right of patronage, and not the other way around.\nPeople therefore also lived there too.\nAs is apparent from the upper administrative office for Freudenstadt's description in 1858, the \"mountain\" was, alongside the lower village and the upper village that can just about be made out still in the appearance of Tumlingen today, the village's third, grand centre of settlement.
Association wants to have wind turbine approval examined\n\"Boisheim wehrt sich\", sees no legal basis for the construction and has therefore turned to the District Administrator.\nThe association \"Boisheim wehrt sich\" has called on District Administrator Andreas Coenen (CDU) to retract his approval for the construction of planned wind turbines in Boisheimer Nette.\nThey say it was unlawful.\nThe association argues that at the time the approval was given on 30th December 2016, no wind turbines could have been approved in Bosheimer Nette, since at that time only one wind energy concentration zone existed in Viersen, on the city boundary with Schwalmtal.\nConstruction and operation of wind energy plants are in principle possible within this concentration zone, but at the same time, the concentration zone has an exclusion against the use of wind energy for the rest of the city area.\nSuch zones are said to prevent \"asparagusisation\" of the landscape.\nThe city of Viersen had in fact passed a resolution for a second wind energy concentration zone in Boisheimer Nette a few days earlier, through an amendment to the local land-use plan \u2013 but the D\u00fcsseldorf regional government had not approved it due to a formal error.\n\"Due to the lack of legal planning requirements, the approval decision that was issued is still unlawful, and should therefore be retracted as well,\" said Christoph Erkens, chairman of \"Boisheim wehrt sich\".\nThe current state of affairs: the city council once again passed a resolution for the wind concentration zone in Boisheimer Nette before the summer break; approval from the regional government is still pending.

User Instruction 3: Here’s a new batch.
Docs 5-7: Town and dealers plan \"hometown shopping\" weekend\nBad Langensalzaer and people from the region are to newly discover the shopping opportunities in their town centre in September.\nThe trade association and town authorities want to use the weekend of the 8th and 9th September to entice residents to a shopping spree in the town centre.\nAs chairman of the trade association Christian K\u00f6nig announced, the \"home-town shopping\" weekend is meant to make customers aware of what offers their own town has available aside from the usual supermarket goods.\nA market with local suppliers is thus planned, similar to the weekly market.\nIn addition, homegrown companies are to give showcases, with their apprenticeship places.\nFurthermore, the businesses in the town centre are to invite people to come \"midnight shopping\".\nThe campaign is not only for residents of the town and the nearby region.\nAdverts on the radio, an online video and printed bags that allude to the campaign weekend are in preparation, Christian K\u00f6nig tells us.\nTown authorities and the trade association are cooperating with the Chamber of Industry and Commerce.
Max Menge works as volunteer\nDuring your voluntary service year, you're working at many locations simultaneously.\nWhat are all the different places?\nMenge: Normally in Steinfurt in the furniture exchange, and once a week I help out here in Greven in the M\u00f6belbr\u00fccke.\nAnd I have also worked at the full-day school in their supplementary activities.\nWhat are your experiences of that?\nMenge: It is a nice, broad spectrum that I see here.\nI deal with different people: with adolescents, with refugees, with young adults.\nYou are 26 years old now, so a volunteer with a late-calling.\nHow did that come about?
Municipal council wants common room\nIn the end, the decision was clear: with two dissenting votes, the municipal council of Fluorn-Winzeln passed a resolution for a common room in the future retirement home.\nAhead of the debate, there is an important piece of information to be had: the purchase contract for the as yet still needed plot section has been signed, reports mayor Bernhard Tjaden.\nAdmittedly it will still be a while until the plot, which up until now has belonged to the church parish, comes into the possession of the municipality, but nevertheless, there are now no longer any planning obstacles for the project that has to be materialised by a private investor.\nThat is why there is a debate concerning a common room.\nOne of the two investors that are still on board has offered to accommodate the municipality.\nFor a special price, it would provide a living space \u2013 of around 55 square metres \u2013 and an adjacent terrace with no less than 70 square metres on the ground floor.\nWhat the management and the majority of the municipal council have in mind is not simply to set aside such a common room as a meeting place only for the residents of the senior citizens' accommodation, but for it to be open to the community.\nTalk of a \"knitting caf\u00e9\" was making the rounds, for example.\nThe decision, which was already pre-determined last Autumn through the allocation of resources for the current budget, is sweetened by funding opportunities.

User Instruction 4: I've got one more.
Doc 8: Home connections prove to be very complex\nThe building works in Villinger Stra\u00dfe are in full throw and \"are more or less on schedule,\" according to the manager of the construction yard, Fred Heinze.\nIn terms of home connections, the situation is more complicated than expected, since lots of things are hidden in the substrate that were not noted in the plans.\nThe traffic, which always disrupts building works, poses a big problem for the Stumpp company construction crew.\n\"The ignorance of car drivers around red lights and barriers is really huge,\" reports Heinze.\nThe street is closed off and only the public busses are allowed to drive through.\nAngelo Caruso, Marvin Kostorz and Tim Kostorz from the Stumpp company have been engaged in the works on the bridges for weeks.\n\"The hot days were pretty exhausting, for sure.\nBut luckily we were supplied with cold mineral water by our friend Fritz Matt,\" explained Marvin Kostorz.\n
"""
        )
    },
    {
        "role": "assistant",
        "content": 
"""Write a summary about the following 8 documents, on the query: What are the names of the people, cities, and organizations who are involved in construction efforts?

Documents:
Munich 1856: Four maps that will change your view of the city\nA mental asylum, where today young people are said to meet.\nA crypt chapel, where they are now digging tunnels for the S-Bahn.\nAllotment holders cultivate the soil of former farmers.\nThe oldest official map of Munich brings captivating stories to light.\nIt is annoying when geographical maps are not up-to-date.\nAnyone who has ever got worked up because the car's sat-nav is showing a green field instead of a bypass knows that.\nThe historical maps of the digital BayernAtlas, an offering from the State Government's Geoportal Bayern, are anything but up-to-date \u2013 and yet it is precisely for this reason that they are so informative.\nEspecially when one compares them with current online maps.\nThen it becomes clear how the towns and municipalities in the distribution area of Munich's Merkur newspaper have changed since the 19th century.
Pasture fence project is fundamental\nThe Fischerbach pasture fence project is a successful project and will be continued next year.\nMembers of the Bundestag for the CDU Peter Wei\u00df and Kordula Kovac got a first-hand view.\nRepresentatives from the Fischerbach local CDU group met with mayor Thomas Schneider and the two representatives at Rechtgrabenhof farm, belonging to the Schwendemann family.\nTo start with, Markus Schwendemann gave insights into his business, in which he cultivates 55 hectares of pasture and 40 hectares of woodland, keeps 40 milk cows and just as many young stock, and accommodates five apartments of holiday makers.\nSchwendemann says that, for him, there is no alternative to the \"Black Forest milk\" marketing channel he uses for the milk from his cows.\nOnly regional produce is served to the holiday-makers.\nSince he is also BLHV chairman for Fischerbach, he described the pasture fence project as \"fundamentally important\".\nI can't make a profit on that kind of steep-sloping ground.\nGrassland areas make for a lot of work, a lot has to be done with the string trimmer.
Tumlingen is celebrating 750 years of the St. Hilarius Church\n750 years of the St. Hilarius Church: To mark the anniversary of the mountain church, invitations are being issued to a serenade in the old cemetery grounds in Tumlingen on Sunday 16th July at 4.30pm.\nDue to its idyllic setting and beautiful view over the village, the historical site of St. Hilarius is a beloved place to tarry a while.\nVisitors to the sixth benefit serenade can let themselves be enchanted by this powerful place.\nThe Friends and Supporters of Tumlingen have made the rescue and preservation of the oldest building in the area their concern.\nWalter Martini stresses: \"It is a one-of-a-kind rural jewel\".\nThe St. Hilarius chapel was first mentioned in records in 1267 in the W\u00fcrttemberg register: Count Hermann von Sulz sold \"the mountain with the parish church for Tumlingen on it, together with the Bailiwick\" to Master Eberhard von Horb and his brother Dietrich for 20 silver marks.\nAccording to the text of the official document, aside from the former parish church, a lordly farm also stood on the Tumlingen mountain, which itself owned the church, or the right of patronage, and not the other way around.\nPeople therefore also lived there too.\nAs is apparent from the upper administrative office for Freudenstadt's description in 1858, the \"mountain\" was, alongside the lower village and the upper village that can just about be made out still in the appearance of Tumlingen today, the village's third, grand centre of settlement.
Association wants to have wind turbine approval examined\n\"Boisheim wehrt sich\", sees no legal basis for the construction and has therefore turned to the District Administrator.\nThe association \"Boisheim wehrt sich\" has called on District Administrator Andreas Coenen (CDU) to retract his approval for the construction of planned wind turbines in Boisheimer Nette.\nThey say it was unlawful.\nThe association argues that at the time the approval was given on 30th December 2016, no wind turbines could have been approved in Bosheimer Nette, since at that time only one wind energy concentration zone existed in Viersen, on the city boundary with Schwalmtal.\nConstruction and operation of wind energy plants are in principle possible within this concentration zone, but at the same time, the concentration zone has an exclusion against the use of wind energy for the rest of the city area.\nSuch zones are said to prevent \"asparagusisation\" of the landscape.\nThe city of Viersen had in fact passed a resolution for a second wind energy concentration zone in Boisheimer Nette a few days earlier, through an amendment to the local land-use plan \u2013 but the D\u00fcsseldorf regional government had not approved it due to a formal error.\n\"Due to the lack of legal planning requirements, the approval decision that was issued is still unlawful, and should therefore be retracted as well,\" said Christoph Erkens, chairman of \"Boisheim wehrt sich\".\nThe current state of affairs: the city council once again passed a resolution for the wind concentration zone in Boisheimer Nette before the summer break; approval from the regional government is still pending.
Town and dealers plan \"hometown shopping\" weekend\nBad Langensalzaer and people from the region are to newly discover the shopping opportunities in their town centre in September.\nThe trade association and town authorities want to use the weekend of the 8th and 9th September to entice residents to a shopping spree in the town centre.\nAs chairman of the trade association Christian K\u00f6nig announced, the \"home-town shopping\" weekend is meant to make customers aware of what offers their own town has available aside from the usual supermarket goods.\nA market with local suppliers is thus planned, similar to the weekly market.\nIn addition, homegrown companies are to give showcases, with their apprenticeship places.\nFurthermore, the businesses in the town centre are to invite people to come \"midnight shopping\".\nThe campaign is not only for residents of the town and the nearby region.\nAdverts on the radio, an online video and printed bags that allude to the campaign weekend are in preparation, Christian K\u00f6nig tells us.\nTown authorities and the trade association are cooperating with the Chamber of Industry and Commerce.
Max Menge works as volunteer\nDuring your voluntary service year, you're working at many locations simultaneously.\nWhat are all the different places?\nMenge: Normally in Steinfurt in the furniture exchange, and once a week I help out here in Greven in the M\u00f6belbr\u00fccke.\nAnd I have also worked at the full-day school in their supplementary activities.\nWhat are your experiences of that?\nMenge: It is a nice, broad spectrum that I see here.\nI deal with different people: with adolescents, with refugees, with young adults.\nYou are 26 years old now, so a volunteer with a late-calling.\nHow did that come about?
Municipal council wants common room\nIn the end, the decision was clear: with two dissenting votes, the municipal council of Fluorn-Winzeln passed a resolution for a common room in the future retirement home.\nAhead of the debate, there is an important piece of information to be had: the purchase contract for the as yet still needed plot section has been signed, reports mayor Bernhard Tjaden.\nAdmittedly it will still be a while until the plot, which up until now has belonged to the church parish, comes into the possession of the municipality, but nevertheless, there are now no longer any planning obstacles for the project that has to be materialised by a private investor.\nThat is why there is a debate concerning a common room.\nOne of the two investors that are still on board has offered to accommodate the municipality.\nFor a special price, it would provide a living space \u2013 of around 55 square metres \u2013 and an adjacent terrace with no less than 70 square metres on the ground floor.\nWhat the management and the majority of the municipal council have in mind is not simply to set aside such a common room as a meeting place only for the residents of the senior citizens' accommodation, but for it to be open to the community.\nTalk of a \"knitting caf\u00e9\" was making the rounds, for example.\nThe decision, which was already pre-determined last Autumn through the allocation of resources for the current budget, is sweetened by funding opportunities.
Home connections prove to be very complex\nThe building works in Villinger Stra\u00dfe are in full throw and \"are more or less on schedule,\" according to the manager of the construction yard, Fred Heinze.\nIn terms of home connections, the situation is more complicated than expected, since lots of things are hidden in the substrate that were not noted in the plans.\nThe traffic, which always disrupts building works, poses a big problem for the Stumpp company construction crew.\n\"The ignorance of car drivers around red lights and barriers is really huge,\" reports Heinze.\nThe street is closed off and only the public busses are allowed to drive through.\nAngelo Caruso, Marvin Kostorz and Tim Kostorz from the Stumpp company have been engaged in the works on the bridges for weeks.\n\"The hot days were pretty exhausting, for sure.\nBut luckily we were supplied with cold mineral water by our friend Fritz Matt,\" explained Marvin Kostorz.\n
"""
    },
]

Summary_system_claude = "You are a prompt rewriter whose main goal is to rewrite the prompt given by the user in the most optimal way without losing any information in them."
"""Eval set v5 generator — expanded, balanced mirror of the 8 official Track 1 categories.

Why v5 exists: v4 (66 tasks, ~8/category) was too small to estimate local accuracy or the
escalation ladder reliably — the point-biserial correlations and per-category rates were dominated
by 1-2 datapoints (see docs/escalation-math.md §9). v5 roughly doubles the set (~16/category, 124
total) and deliberately over-samples the *trap* cases where a small local model is confidently
wrong (math word-problems, deductive riddles, factual hallucinations) — exactly the tasks the
escalation policy has to catch.

Each task keeps the most deterministic grader possible:
  - exact         : one of the answer strings appears (boundary-aware, see agent_eval._norm_match)
  - contains_all  : ALL required strings appear (NER, multi-part factual)
  - pytests       : extract code from the answer, run it + assert (code gen/debug)
  - judge         : an independent LLM (gemma3:12b) vs a reference (summarisation only)

Code tasks additionally carry a `solution` field — a known-correct implementation. It is NOT used
by the agent or the LLM eval; it lets eval/validate_tasks.py *prove offline* that every `tests`
block is satisfiable (a wrong test would silently poison the accuracy estimate).

Run:  python -m eval.gen_tasks_v5   ->   eval/tasks_v5.jsonl
Then: python -m eval.validate_tasks --tasks eval/tasks_v5.jsonl   (LLM-free, run this first)
Then: OLLAMA_HOST=... python -m eval.agent_eval --tasks eval/tasks_v5.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).parent / "tasks_v5.jsonl"

TASKS: list[dict] = []


def add(category, prompt, grader, *, answer=None, reference=None, tests=None, solution=None):
    t = {"id": len(TASKS) + 1, "category": category, "input": prompt, "grader": grader}
    if answer is not None:
        t["answer"] = answer if isinstance(answer, list) else [str(answer)]
    if reference is not None:
        t["reference"] = reference
    if tests is not None:
        t["tests"] = tests
    if solution is not None:
        t["solution"] = solution
    TASKS.append(t)


# =================================================================== factual (16)
add("factual", "What is the capital of Australia, and what body of water is it near?",
    "contains_all", answer=["Canberra", "Burley Griffin"])
add("factual", "What is the chemical symbol for gold?", "exact", answer=["Au"])
add("factual", "Which planet in our solar system is known as the Red Planet?", "exact", answer=["Mars"])
add("factual", "Who wrote the novel Nineteen Eighty-Four?", "exact", answer=["Orwell"])
add("factual", "What is the largest ocean on Earth?", "exact", answer=["Pacific"])
add("factual", "In what year did the Berlin Wall fall?", "exact", answer=["1989"])
add("factual", "Which chemical element has atomic number 6?", "exact", answer=["carbon"])
add("factual", "What is the longest river in South America?", "exact", answer=["Amazon"])
add("factual", "What gas do plants primarily absorb from the air during photosynthesis?",
    "exact", answer=["carbon dioxide", "CO2"])
add("factual", "Which country hosted the first modern Olympic Games in 1896, and in which city?",
    "contains_all", answer=["Greece", "Athens"])
add("factual", "What is the capital of Japan?", "exact", answer=["Tokyo"])
add("factual", "What is the chemical symbol for sodium?", "exact", answer=["Na"])
add("factual", "Who painted the Mona Lisa?", "exact", answer=["da Vinci", "Leonardo"])
add("factual", "What is the tallest mountain on Earth above sea level?", "exact", answer=["Everest"])
add("factual", "Which planet is closest to the Sun?", "exact", answer=["Mercury"])
add("factual", "How many continents are there on Earth?", "exact", answer=["7", "seven"])

# =================================================================== math (16)
add("math", "A store has 320 items. It sells 25% of them on Monday and 40 more on Tuesday. "
    "How many items remain?", "exact", answer=["200"])
add("math", "A jacket costs $150. It is discounted 20%, then an additional 10% off the reduced "
    "price. What is the final price in dollars?", "exact", answer=["108"])
add("math", "A train travels 180 km in 2.5 hours. What is its average speed in km/h?",
    "exact", answer=["72"])
add("math", "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
    "How much does the ball cost in dollars?", "exact", answer=["0.05", "$0.05", ".05", "5 cents"])
add("math", "What is 17 multiplied by 24?", "exact", answer=["408"])
add("math", "What is the sum of all integers from 1 to 40?", "exact", answer=["820"])
add("math", "What is the greatest common divisor of 84 and 36?", "exact", answer=["12"])
add("math", "A snail climbs a 12-meter well: it climbs 4 meters each day and slips back 3 meters "
    "each night. On which day does it reach the top?", "exact", answer=["9", "ninth"])
add("math", "If it takes 5 machines 5 minutes to make 5 widgets, how many minutes would it take "
    "100 machines to make 100 widgets?", "exact", answer=["5", "five"])
add("math", "What is 15% of 240?", "exact", answer=["36"])
add("math", "A shirt originally costs $80 and is on sale for 25% off. What is the sale price in "
    "dollars?", "exact", answer=["60"])
add("math", "What is the sum of all even numbers from 1 to 20 inclusive?", "exact", answer=["110"])
add("math", "If 3 pens cost $6, how much do 7 pens cost, in dollars?", "exact", answer=["14"])
add("math", "A clock takes 5 seconds to strike 6 o'clock (6 chimes). How many seconds does it "
    "take to strike 12 o'clock (12 chimes)?", "exact", answer=["11"])
add("math", "A rectangle is 8 cm long and 5 cm wide. What is its area in square centimeters?",
    "exact", answer=["40"])
add("math", "There are 5 apples on a table. You take away 3. How many apples do you have?",
    "exact", answer=["3", "three"])

# =================================================================== sentiment (16)
add("sentiment", "Classify the sentiment of this review: The food was absolutely delicious and "
    "the staff were wonderful.", "exact", answer=["positive"])
add("sentiment", "Classify the sentiment of this review: Terrible service, cold food, and we "
    "waited an hour. Never coming back.", "exact", answer=["negative"])
add("sentiment", "Classify the sentiment of this statement: The package arrived on the scheduled "
    "delivery date.", "exact", answer=["neutral"])
add("sentiment", "Classify the sentiment of this review: The battery life is great, but the "
    "screen scratches far too easily.", "exact", answer=["mixed", "neutral"])
add("sentiment", "Classify the sentiment of this review: I was skeptical at first, but this "
    "vacuum exceeded every expectation. Worth every penny!", "exact", answer=["positive"])
add("sentiment", "Classify the sentiment of this tweet: Flight delayed four hours, lost my "
    "luggage, and customer service hung up on me.", "exact", answer=["negative"])
add("sentiment", "Classify the sentiment of this sentence: The meeting has been moved from "
    "3 PM to 4 PM on Thursday.", "exact", answer=["neutral"])
add("sentiment", "Classify the sentiment of this review: Gorgeous design and blazing fast, "
    "though the price stings a little.", "exact", answer=["positive", "mixed"])
add("sentiment", "Classify the sentiment of this review: Hands down the best coffee I have ever "
    "had. I'll be back every morning.", "exact", answer=["positive"])
add("sentiment", "Classify the sentiment of this review: The hotel room was filthy and the "
    "air conditioning was broken the whole trip.", "exact", answer=["negative"])
add("sentiment", "Classify the sentiment of this note: The store is open from 9 AM to 6 PM on "
    "weekdays.", "exact", answer=["neutral"])
add("sentiment", "Classify the sentiment of this review: Fast shipping and the product works, but "
    "the instructions were confusing and incomplete.", "exact", answer=["mixed", "neutral"])
add("sentiment", "Classify the sentiment of this comment: Absolutely thrilled with my new phone, "
    "it's everything I hoped for!", "exact", answer=["positive"])
add("sentiment", "Classify the sentiment of this review: Waste of money. It broke after two days "
    "and support ignored me.", "exact", answer=["negative"])
add("sentiment", "Classify the sentiment of this sentence: The report contains four sections and "
    "two appendices.", "exact", answer=["neutral"])
add("sentiment", "Classify the sentiment of this review: The acting was superb, but the plot "
    "dragged and the ending was disappointing.", "exact", answer=["mixed", "neutral"])

# =================================================================== summarisation (12, judge)
add("summarisation", "Summarize the following in exactly one sentence: The new library opened "
    "downtown last week. It features a rooftop garden, a children's wing, and a cafe. Residents "
    "have praised its modern design and free public workshops.", "judge",
    reference="The new downtown library, which opened last week with a rooftop garden, "
              "children's wing, and cafe, has been praised by residents for its modern design "
              "and free workshops.")
add("summarisation", "Summarize the following in exactly one sentence: Scientists discovered a "
    "new species of deep-sea fish near the Mariana Trench. The fish uses bioluminescence to "
    "attract prey. It can survive extreme pressure at depths of over 8,000 meters.", "judge",
    reference="Scientists discovered a new bioluminescent deep-sea fish near the Mariana Trench "
              "that survives at depths over 8,000 meters.")
add("summarisation", "Summarize the following in one sentence: The city council voted to expand "
    "the bike lane network by 40 kilometers. Construction begins in March and should finish by "
    "October. Local businesses expressed concerns about parking, but cycling groups celebrated "
    "the decision.", "judge",
    reference="The city council approved a 40 km bike lane expansion running March to October, "
              "welcomed by cyclists but worrying businesses about parking.")
add("summarisation", "Summarize the following in exactly two sentences: Remote work adoption "
    "surged during the pandemic and has remained high. Many companies now offer hybrid "
    "schedules. Studies show productivity stayed stable or improved, though some managers worry "
    "about collaboration and mentoring of junior staff.", "judge",
    reference="Remote work rose sharply during the pandemic and remains common, with many firms "
              "adopting hybrid schedules. Productivity has held steady or improved, although "
              "managers still worry about collaboration and mentoring.")
add("summarisation", "Summarize the following in one sentence: A regional airline announced it "
    "will retire its aging turboprop fleet next year. The planes will be replaced with newer, "
    "more fuel-efficient jets. The transition is expected to cut fuel costs by 30 percent and "
    "reduce noise complaints from residents near the airport.", "judge",
    reference="A regional airline will replace its old turboprops with fuel-efficient jets next "
              "year, cutting fuel costs 30% and reducing noise.")
add("summarisation", "Condense this into a single sentence: The museum's new exhibit features "
    "artifacts from ancient Mesopotamia, including clay tablets with cuneiform writing. Curators "
    "spent five years assembling the collection from loans by twelve institutions.", "judge",
    reference="The museum's new ancient-Mesopotamia exhibit, five years in the making with loans "
              "from twelve institutions, features artifacts including cuneiform clay tablets.")
add("summarisation", "Summarize the following in one sentence: A startup unveiled a solar-powered "
    "water purifier for rural areas. It can filter 500 liters a day and needs no grid electricity. "
    "Field trials in three villages cut waterborne illness by half.", "judge",
    reference="A startup's off-grid solar water purifier filters 500 liters a day and halved "
              "waterborne illness in three village field trials.")
add("summarisation", "Summarize the following in exactly one sentence: The national park reopened "
    "its main trail after two years of repairs following a landslide. Rangers added new drainage "
    "and reinforced three bridges. Visitor numbers are expected to rebound this summer.", "judge",
    reference="The national park reopened its main trail after two years of landslide repairs "
              "including new drainage and three reinforced bridges, with visitors expected to "
              "rebound this summer.")
add("summarisation", "Condense this into one sentence: A university team built a low-cost "
    "prosthetic hand that responds to muscle signals. It is 3D-printed and costs under $200. "
    "The design is open-source so clinics anywhere can reproduce it.", "judge",
    reference="A university team created an open-source, 3D-printed muscle-controlled prosthetic "
              "hand costing under $200 that any clinic can reproduce.")
add("summarisation", "Summarize the following in one sentence: The tech conference drew 12,000 "
    "attendees this year, double last year's figure. Keynotes focused on energy-efficient AI "
    "hardware. Several startups announced chips aimed at cutting data-center power use.", "judge",
    reference="This year's tech conference doubled to 12,000 attendees, with keynotes and startup "
              "chip announcements centered on energy-efficient AI hardware.")
add("summarisation", "Summarize the following in exactly two sentences: A coastal town installed "
    "a network of sensors to give early warning of flooding. The system texts residents when water "
    "levels rise. During a storm last month it gave two hours of warning, allowing a smooth "
    "evacuation.", "judge",
    reference="A coastal town's new sensor network texts residents when water levels rise to warn "
              "of flooding. During a storm last month it provided two hours of warning that "
              "enabled a smooth evacuation.")
add("summarisation", "Condense this into a single sentence: The bakery chain switched all its "
    "packaging to compostable materials. It also began sourcing flour from local farms. Sales rose "
    "8 percent as customers responded to the greener image.", "judge",
    reference="After switching to compostable packaging and locally sourced flour, the bakery "
              "chain saw sales rise 8 percent on its greener image.")

# =================================================================== ner (16)
add("ner", "Extract all named entities and their types from: Maria Sanchez joined Fireworks AI "
    "in Berlin last March.", "contains_all", answer=["Maria Sanchez", "Fireworks AI", "Berlin", "March"])
add("ner", "Extract all named entities and their types from: Tim Cook announced the partnership "
    "at Apple Park in Cupertino on Tuesday.", "contains_all",
    answer=["Tim Cook", "Apple Park", "Cupertino", "Tuesday"])
add("ner", "Extract the named entities and their types from: Dr. Amara Okafor presented her "
    "research at Oxford University in September.", "contains_all",
    answer=["Amara Okafor", "Oxford University", "September"])
add("ner", "Identify all named entities and their types in: Toyota opened a new plant in "
    "Guadalajara, Mexico, creating 3,000 jobs.", "contains_all",
    answer=["Toyota", "Guadalajara", "Mexico"])
add("ner", "Extract all named entities and their types from: The treaty was signed by Chancellor "
    "Weber and President Silva in Geneva on 12 June 2024.", "contains_all",
    answer=["Weber", "Silva", "Geneva", "12 June 2024"])
add("ner", "Extract named entities and their types from: Lionel Messi scored twice as Inter "
    "Miami beat Orlando City on Saturday.", "contains_all",
    answer=["Lionel Messi", "Inter Miami", "Orlando City", "Saturday"])
add("ner", "List all named entities with their types from: Samsung unveiled the Galaxy Fold at "
    "its headquarters in Seoul during January.", "contains_all",
    answer=["Samsung", "Galaxy Fold", "Seoul", "January"])
add("ner", "Extract all named entities and their types from: NASA launched the Artemis mission "
    "from Kennedy Space Center in Florida.", "contains_all",
    answer=["NASA", "Artemis", "Kennedy Space Center", "Florida"])
add("ner", "Extract all named entities and their types from: Serena Williams retired after "
    "beating Anna Kova at the US Open in New York.", "contains_all",
    answer=["Serena Williams", "Anna Kova", "US Open", "New York"])
add("ner", "Identify all named entities and their types in: Microsoft acquired GitHub for "
    "$7.5 billion in 2018.", "contains_all", answer=["Microsoft", "GitHub", "2018"])
add("ner", "Extract named entities and their types from: Angela Merkel met Emmanuel Macron in "
    "Paris on Monday.", "contains_all", answer=["Angela Merkel", "Emmanuel Macron", "Paris", "Monday"])
add("ner", "List all named entities with their types from: The Amazon River flows through Brazil "
    "and Peru into the Atlantic Ocean.", "contains_all",
    answer=["Amazon River", "Brazil", "Peru", "Atlantic Ocean"])
add("ner", "Extract all named entities and their types from: Pixar released Toy Story in "
    "California in November 1995.", "contains_all",
    answer=["Pixar", "Toy Story", "California", "November 1995"])
add("ner", "Extract all named entities and their types from: Elena Rossi flew from Rome to Tokyo "
    "with Alitalia in April.", "contains_all", answer=["Elena Rossi", "Rome", "Tokyo", "Alitalia", "April"])
add("ner", "Identify all named entities and their types in: Google opened an office in Nairobi, "
    "Kenya, led by Joseph Mwangi.", "contains_all",
    answer=["Google", "Nairobi", "Kenya", "Joseph Mwangi"])
add("ner", "Extract named entities and their types from: The Louvre in Paris displayed a painting "
    "by Frida Kahlo last spring.", "contains_all", answer=["Louvre", "Paris", "Frida Kahlo"])

# =================================================================== code_debug (16)
add("code_debug", "This function should return the max of a list but has a bug: "
    "def get_max(nums): return nums[0]. Find and fix it.", "pytests",
    solution="def get_max(nums):\n    return max(nums)",
    tests="assert get_max([3,1,2])==3\nassert get_max([-5,-2,-9])==-2\nassert get_max([7])==7")
add("code_debug", "This function should sum the integers from 1 to n inclusive but has a bug: "
    "def sum_to_n(n): return sum(range(n)). Fix it.", "pytests",
    solution="def sum_to_n(n):\n    return sum(range(1, n+1))",
    tests="assert sum_to_n(5)==15\nassert sum_to_n(1)==1\nassert sum_to_n(10)==55")
add("code_debug", "This function should return True when n is even, but it is wrong: "
    "def is_even(n): return n % 2 == 1. Fix it.", "pytests",
    solution="def is_even(n):\n    return n % 2 == 0",
    tests="assert is_even(4) is True\nassert is_even(7) is False\nassert is_even(0) is True")
add("code_debug", "This function should reverse a string but has a bug: "
    "def reverse(s): return s[::2]. Fix it.", "pytests",
    solution="def reverse(s):\n    return s[::-1]",
    tests="assert reverse('abc')=='cba'\nassert reverse('hello')=='olleh'\nassert reverse('')==''")
add("code_debug", "This function should return the average as a float but has a bug: "
    "def average(nums): return sum(nums) // len(nums). Fix it.", "pytests",
    solution="def average(nums):\n    return sum(nums) / len(nums)",
    tests="assert average([1,2])==1.5\nassert average([3,3,3])==3.0")
add("code_debug", "This recursive factorial has a bug: "
    "def fact(n):\n    if n == 1: return 0\n    return n * fact(n-1)\nFix it.", "pytests",
    solution="def fact(n):\n    if n <= 1: return 1\n    return n * fact(n-1)",
    tests="assert fact(1)==1\nassert fact(5)==120\nassert fact(3)==6")
add("code_debug", "This function should count how many times x appears in a list but is buggy: "
    "def count_x(lst, x): return len(lst) - lst.count(x). Fix it.", "pytests",
    solution="def count_x(lst, x):\n    return lst.count(x)",
    tests="assert count_x([1,2,2,3],2)==2\nassert count_x([],5)==0\nassert count_x([4,4,4],4)==3")
add("code_debug", "This function should return the first word of a sentence but has a bug: "
    "def first_word(s): return s.split()[-1]. Fix it.", "pytests",
    solution="def first_word(s):\n    return s.split()[0]",
    tests="assert first_word('hello world')=='hello'\nassert first_word('one')=='one'")
add("code_debug", "This function should return True only for positive numbers, but it is wrong: "
    "def is_positive(n): return n >= 0. Fix it.", "pytests",
    solution="def is_positive(n):\n    return n > 0",
    tests="assert is_positive(5) is True\nassert is_positive(0) is False\nassert is_positive(-3) is False")
add("code_debug", "This function should convert Celsius to Fahrenheit but has a bug: "
    "def c_to_f(c): return c * 9/5. Fix it.", "pytests",
    solution="def c_to_f(c):\n    return c * 9/5 + 32",
    tests="assert c_to_f(0)==32\nassert c_to_f(100)==212\nassert c_to_f(37)==98.6")
add("code_debug", "This function should return the last element of a list but has a bug: "
    "def last(lst): return lst[1]. Fix it.", "pytests",
    solution="def last(lst):\n    return lst[-1]",
    tests="assert last([1,2,3])==3\nassert last([9])==9\nassert last(['a','b'])=='b'")
add("code_debug", "This function should double every number in a list but has a bug: "
    "def double_list(lst): return [x+2 for x in lst]. Fix it.", "pytests",
    solution="def double_list(lst):\n    return [x*2 for x in lst]",
    tests="assert double_list([1,2,3])==[2,4,6]\nassert double_list([])==[]\nassert double_list([0])==[0]")
add("code_debug", "This function should count the words in a sentence but has a bug: "
    "def count_words(s): return len(s). Fix it.", "pytests",
    solution="def count_words(s):\n    return len(s.split())",
    tests="assert count_words('a b c')==3\nassert count_words('hello')==1\nassert count_words('one two')==2")
add("code_debug", "This function should return the larger of two numbers but has a bug: "
    "def max_two(a, b): return a. Fix it.", "pytests",
    solution="def max_two(a, b):\n    return a if a > b else b",
    tests="assert max_two(3,7)==7\nassert max_two(9,2)==9\nassert max_two(4,4)==4")
add("code_debug", "This function should remove all spaces from a string but has a bug: "
    "def strip_spaces(s): return s.strip(). Fix it.", "pytests",
    solution="def strip_spaces(s):\n    return s.replace(' ', '')",
    tests="assert strip_spaces('a b c')=='abc'\nassert strip_spaces('  x  ')=='x'\nassert strip_spaces('no')=='no'")
add("code_debug", "This function should return the number of items greater than 10 in a list but "
    "is buggy: def count_big(lst): return sum(1 for x in lst if x < 10). Fix it.", "pytests",
    solution="def count_big(lst):\n    return sum(1 for x in lst if x > 10)",
    tests="assert count_big([5,12,20,3])==2\nassert count_big([1,2,3])==0\nassert count_big([11,11])==2")

# =================================================================== logical (16)
add("logical", "Three friends, Ana, Ben, and Carl, each own a different pet: cat, dog, bird. "
    "Ana does not own the bird. Ben owns the dog. Who owns the cat?", "exact", answer=["Ana"])
add("logical", "All roses are flowers. Some flowers fade quickly. Can we logically conclude "
    "that some roses fade quickly? Answer yes or no.", "exact", answer=["no"])
add("logical", "Sally has 3 brothers. Each brother has 2 sisters. How many sisters does Sally "
    "have?", "exact", answer=["1", "one"])
add("logical", "There are three killers in a room. Someone enters and kills one of them. Nobody "
    "leaves the room. How many killers are left in the room?", "exact", answer=["3", "three"])
add("logical", "Tom is taller than Jim. Jim is taller than Sue. Who is the shortest?",
    "exact", answer=["Sue"])
add("logical", "A farmer has 17 sheep. All but 9 run away. How many sheep are left?",
    "exact", answer=["9", "nine"])
add("logical", "If two days after tomorrow is Friday, what day is today?", "exact",
    answer=["Tuesday"])
add("logical", "On an island, knights always tell the truth and knaves always lie. A person "
    "says: 'I am a knave.' Is this possible? Answer yes or no.", "exact", answer=["no"])
add("logical", "All Bloops are Razzies and all Razzies are Lazzies. Are all Bloops definitely "
    "Lazzies? Answer yes or no.", "exact", answer=["yes"])
add("logical", "Alex is heavier than Bob. Carol is lighter than Bob. Who is the heaviest?",
    "exact", answer=["Alex"])
add("logical", "Mary's father has five daughters: Nana, Nene, Nini, Nono, and one more. What is "
    "the fifth daughter's name?", "exact", answer=["Mary"])
add("logical", "If yesterday was Sunday, what day will it be tomorrow?", "exact", answer=["Tuesday"])
add("logical", "In a race, you overtake the person in second place. What position are you in now?",
    "exact", answer=["second", "2nd"])
add("logical", "All cars have wheels. A bicycle has wheels. Does it follow that a bicycle is a "
    "car? Answer yes or no.", "exact", answer=["no"])
add("logical", "All birds have feathers. A penguin is a bird. Does a penguin have feathers? "
    "Answer yes or no.", "exact", answer=["yes"])
add("logical", "A box contains only red and blue balls. Every red ball is heavier than every blue "
    "ball. Ball X is lighter than ball Y. If exactly one of them is red, which one is red?",
    "exact", answer=["Y"])

# =================================================================== code_gen (16)
add("code_gen", "Write a Python function named second_largest(nums) that returns the "
    "second-largest distinct number in a list, handling duplicates correctly.", "pytests",
    solution="def second_largest(nums):\n    return sorted(set(nums))[-2]",
    tests="assert second_largest([1,2,3])==2\nassert second_largest([5,5,4])==4\n"
          "assert second_largest([10,10,10,7])==7")
add("code_gen", "Write a Python function named reverse_words(s) that reverses the order of "
    "words in a string. Words are separated by single spaces.", "pytests",
    solution="def reverse_words(s):\n    return ' '.join(s.split()[::-1])",
    tests="assert reverse_words('hello world')=='world hello'\n"
          "assert reverse_words('a b c')=='c b a'\nassert reverse_words('one')=='one'")
add("code_gen", "Write a Python function named is_palindrome(s) that returns True if the string "
    "is a palindrome, ignoring case.", "pytests",
    solution="def is_palindrome(s):\n    s = s.lower()\n    return s == s[::-1]",
    tests="assert is_palindrome('Racecar') is True\nassert is_palindrome('hello') is False\n"
          "assert is_palindrome('') is True")
add("code_gen", "Write a Python function named fizzbuzz(n) that returns 'Fizz' if n is divisible "
    "by 3, 'Buzz' if divisible by 5, 'FizzBuzz' if divisible by both, otherwise the number as a "
    "string.", "pytests",
    solution="def fizzbuzz(n):\n    if n % 15 == 0: return 'FizzBuzz'\n    if n % 3 == 0: return 'Fizz'\n"
             "    if n % 5 == 0: return 'Buzz'\n    return str(n)",
    tests="assert fizzbuzz(3)=='Fizz'\nassert fizzbuzz(5)=='Buzz'\nassert fizzbuzz(15)=='FizzBuzz'\n"
          "assert fizzbuzz(7)=='7'")
add("code_gen", "Write a Python function named count_vowels(s) that returns the number of vowels "
    "(a, e, i, o, u, case-insensitive) in a string.", "pytests",
    solution="def count_vowels(s):\n    return sum(c.lower() in 'aeiou' for c in s)",
    tests="assert count_vowels('hello')==2\nassert count_vowels('AEIOU')==5\nassert count_vowels('xyz')==0")
add("code_gen", "Write a Python function named flatten(lst) that flattens a list of lists one "
    "level deep into a single list.", "pytests",
    solution="def flatten(lst):\n    return [x for sub in lst for x in sub]",
    tests="assert flatten([[1,2],[3],[4,5]])==[1,2,3,4,5]\nassert flatten([])==[]\n"
          "assert flatten([[],[1]])==[1]")
add("code_gen", "Write a Python function named nth_fib(n) that returns the nth Fibonacci number "
    "iteratively, where nth_fib(0)=0 and nth_fib(1)=1.", "pytests",
    solution="def nth_fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
    tests="assert nth_fib(0)==0\nassert nth_fib(1)==1\nassert nth_fib(10)==55")
add("code_gen", "Write a Python function named are_anagrams(a, b) that returns True if the two "
    "strings are anagrams of each other, ignoring case.", "pytests",
    solution="def are_anagrams(a, b):\n    return sorted(a.lower()) == sorted(b.lower())",
    tests="assert are_anagrams('Listen','Silent') is True\nassert are_anagrams('cat','dog') is False\n"
          "assert are_anagrams('a','a') is True")
add("code_gen", "Write a Python function named sum_list(nums) that returns the sum of a list of "
    "numbers, returning 0 for an empty list.", "pytests",
    solution="def sum_list(nums):\n    return sum(nums)",
    tests="assert sum_list([1,2,3])==6\nassert sum_list([])==0\nassert sum_list([-1,1])==0")
add("code_gen", "Write a Python function named is_prime(n) that returns True if n is a prime "
    "number and False otherwise.", "pytests",
    solution="def is_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5)+1):\n"
             "        if n % i == 0: return False\n    return True",
    tests="assert is_prime(2) is True\nassert is_prime(13) is True\nassert is_prime(1) is False\n"
          "assert is_prime(15) is False")
add("code_gen", "Write a Python function named unique(lst) that returns the distinct elements of "
    "a list, preserving the order of first appearance.", "pytests",
    solution="def unique(lst):\n    seen = []\n    for x in lst:\n        if x not in seen: seen.append(x)\n"
             "    return seen",
    tests="assert unique([1,1,2,3,3])==[1,2,3]\nassert unique([])==[]\nassert unique([5,5,5])==[5]")
add("code_gen", "Write a Python function named gcd(a, b) that returns the greatest common divisor "
    "of two positive integers.", "pytests",
    solution="def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a",
    tests="assert gcd(12,8)==4\nassert gcd(17,5)==1\nassert gcd(100,10)==10")
add("code_gen", "Write a Python function named title_case(s) that capitalizes the first letter of "
    "each word in a string, lowercasing the rest.", "pytests",
    solution="def title_case(s):\n    return ' '.join(w.capitalize() for w in s.split())",
    tests="assert title_case('hello world')=='Hello World'\nassert title_case('a b')=='A B'\n"
          "assert title_case('PYTHON')=='Python'")
add("code_gen", "Write a Python function named count_char(s, c) that returns how many times the "
    "character c appears in the string s.", "pytests",
    solution="def count_char(s, c):\n    return s.count(c)",
    tests="assert count_char('banana','a')==3\nassert count_char('','x')==0\nassert count_char('abc','z')==0")
add("code_gen", "Write a Python function named running_total(nums) that returns a list of the "
    "cumulative sums of the input list.", "pytests",
    solution="def running_total(nums):\n    out, t = [], 0\n    for x in nums:\n        t += x\n"
             "        out.append(t)\n    return out",
    tests="assert running_total([1,2,3])==[1,3,6]\nassert running_total([])==[]\n"
          "assert running_total([5])==[5]")
add("code_gen", "Write a Python function named most_common(lst) that returns the element that "
    "appears most frequently in a non-empty list (any one of the ties is acceptable).", "pytests",
    solution="def most_common(lst):\n    return max(set(lst), key=lst.count)",
    tests="assert most_common([1,2,2,3])==2\nassert most_common(['a','a','b'])=='a'\n"
          "assert most_common([7])==7")


def main():
    from collections import Counter
    with open(OUT, "w") as f:
        for t in TASKS:
            f.write(json.dumps(t) + "\n")
    c = Counter(t["category"] for t in TASKS)
    g = Counter(t["grader"] for t in TASKS)
    print(f"Wrote {len(TASKS)} tasks -> {OUT}")
    print("  per category:", dict(sorted(c.items())))
    print("  per grader  :", dict(sorted(g.items())))


if __name__ == "__main__":
    main()

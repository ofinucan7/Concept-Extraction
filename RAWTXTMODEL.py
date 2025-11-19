import os
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "model1"

# --- PARAMETERS ---
INPUT_DIR = "CS-0007"
OUTPUT_DIR = f"{MODEL}-outputs/outputs-0007"
MODEL = "gpt-5"  # or "gpt-4o-mini" if cost-sensitive

# --- PROMPT TEMPLATE ---
ANNOTATION_PROMPT = """
### ROLE ###
You are an expert academic annotator specializing in *concept extraction* from university lecture slides.

Your goal is to identify and list only the key **concepts** that are *explicitly* defined, emphasized, or used in examples within the provided slide text.

Do NOT infer concepts beyond the slide text. Your task is purely textual and evidence-based.

---

### CODEBOOK (Condensed from “Concept Annotation Codebook – Refined, Oct 2025”) ###

1. **Slide-Deck Grounding Rule** — Only annotate concepts that are justified or defined in the given slide text. Do not infer from outside knowledge.

2. **Granularity Rule** — Annotate at the most specific level that adds distinct meaning.  
   - If splitting a phrase adds no new meaning, keep it as one concept.  
   - If the parts are meaningful alone, include both the whole and its parts.

3. **Definition & Emphasis Rule** — Label any word or phrase that is:
   - Clearly *defined or described*,  
   - *Emphasized* as a key term, or  
   - Used as the *subject* of an example or explanation.

4. **Abbreviation Rule** — Always include abbreviations *and* their expanded forms on separate lines.

5. **Non-Conceptual Modifiers Rule** — Do **not** include syntax, keywords, or modifiers unless the slide explicitly defines or discusses them as concepts (e.g., “static”, “visibility”).

6. **Adjoinment Rule** — Include multi-word terms when they represent a unified idea (e.g., “function header”, “process control block”).

7. **Generic Term Rule** — Do not label generic or context-free terms (e.g., “thing”, “object”) unless defined specifically.

---

### FILTERING STEP ###
After identifying possible concepts, remove any that are **not explicitly**:
- defined,
- emphasized, or
- used in an example or explanation.

Only keep the *most important instructional concepts*.

---

### OUTPUT REQUIREMENTS ###
- Output **each concept on a new line**.  
- Do **not** number them.  
- Do **not** include reasoning, commentary, or explanations.  
- Include abbreviations *and* their full forms as separate lines.  
- Keep consistent capitalization (use lowercase unless the concept is an acronym).
- Do not include REPEATED COURSE NAME as a concept.

Example Output:

function
function name
function body
function header
parameters
return type

EXAMPLE ANNOTATION:

(SLIDE TEXT)

CS 0007: Introduction to Java
Lecture 7
Nathan Ong
University of Pittsburgh
September 22, 2016

FUNCTIONS

Functions
• A function is similar to its mathematical
counterpart
• f(x) = x2, plug in 3, get 9
• Contains several more parts

Java Functions
• Takes in zero or more parameters,
processes them in the function body,
and returns a result
• Imagine going to BestBuyTM and telling
them you want your computer fixed.
You are telling them to run a fixing
function, with your computer being a
parameter. What you get back is your
fixed computer.

You Already Have the Power!
• You already know how to call functions!
Static:
ClassName.functionName(<parameters
>);
Non-static:
objectName.functionName(<parameter
s>);
• How do I make my own?

Function Components

1.
2.
3.
Function 4.
Header 5.

Function
Body 6.

Visibility type (public/protected/private)
static (For now, required)
Return Type
functionName
Parentheses “()”
–
a)
b)
c)

Parameters
Type1 parameterName1
Type2 parameterName2
…

Curly Brackets/Braces “{}”
–

return a value

A Simple Unnecessary
Function
• I want this function to take two doubles
and return their sum.
• Let us go through the list and see what
needs to be incorporated for the
function.

Function Components
1.
2.
3.
4.
5.

We’ll just use public for now.
static (For now, required)
What is the Return Type?
What is an appropriate functionName?
Parentheses “()”
–

What are the parameters?

6. Curly Brackets/Braces “{}”
–
–

What do we do in the Function Body?
What do we return?

Return Type
• What kind of thing are we going to give
back to the function caller?
• “I want this function to take two
doubles and return their sum.”
• The sum of two doubles better be a
double.

Function Name
• The function name should easily
describe what the function does.
• “I want this function to take two
doubles and return their sum.”
• “sum”? Probably not enough detail,
since there are many types that can be
summed.
• sumDoubles

Parameters
• The required input to the function.
• The data that the function needs in
order to properly execute its duties.
• “I want this function to take two
doubles and return their sum.”
• Two doubles.
• Names?

Function Header

//sums two doubles together
public static double sumDoubles
(double addend1, double addend2)

Function Body
…(Function Header)
{
???
}

Function Body
…(Function Header)
{
double sum = addend1 + addend2;
return sum;
}//end method(double,double)

Function Body
…(Function Header)
{
return addend1 + addend2;
}//end method(double,double)

Function Body
public class Functions
{
…(Function)
public static void main(String[] args)
{
double sum = sumDoubles(2.5,3.9);
System.out.println(sum);
}//end method main
}//End class Functions

import java.util.Scanner;
public class AddingMachine
{
…(Function)
public static void main(String[] args)
{
Scanner scan = new Scanner(System.in);
System.out.println("Please enter a number:");
double firstNum = scan.nextDouble();
System.out.println("Please enter a second
number:");
double secondNum = scan.nextDouble();
double sum = sumDoubles(firstNum,secondNum);
System.out.println("The sum of " + firstNum +
" and " + secondNum + " equals " +
sum);
}//end method main
}//End class AddingMachine

Scope
• Why did we need to submit firstNum
and secondNum to sumDoubles?
• Why can’t sumDoubles just use the
already created variables?
• This relates to scope.

Scope
Confidential
Secret
Top Secret

Scope
public class
AddingMachine
public
static void
main

public
static
double
sumDoubl
es

Passing-in Parameters
• The act of providing parameters in a
function call is called Passing-in.
• Java has two kinds of passing-in.

Pass-by-Value
• Passing-in a copy of the value of the
variable.
• Any change made to the variable is not
reflected when the function returns.
• All primitive type parameters are passby-value.

Pass-by-Reference
• Passing-in the actual variable.
• Any change made to the variable IS
reflected when the function returns,
UNLESS the parameter name is
REASSIGNED.
• All non-primitive type parameters are
pass-by-reference.

Another Simple Function
• I want a function that prints a nice
British greeting, complete with the
person’s full name.
Source:
http://www.bbcamerica.com/anglophen
ia/2011/07/five-slang-ways-to-say-hello/

Function Components
1.
2.
3.
4.
5.

public
static
What is the Return Type?
What is an appropriate functionName?
Parentheses “()”
–

What are the parameters?

6. Curly Brackets/Braces “{}”
–
–

What do we do in the Function Body?
What do we return?

Return Type
• Does this function even need to return
any data or information?
• No, so we should use void as our
return type.

Function Header

public static void britishGreeting
(String fullName)

Function Body
…(Function Header)
{
System.out.println("Wotcha " +
fullName +
". Fancy a cuppa?");
}//end method(String)
//Note no return statement



(ENDING CONCEPTS)

Function
Functions
Parameter
Parameters
function body
return
Returns
function header
function name
header
return type
scope
passing-in
pass-by-value
pass-by-reference
void
variable
variables	
Primitive
Public

"""

# --- MAIN PIPELINE ---
def annotate_text_file(file_path):
    with open(file_path, "r") as f:
        slide_text = f.read()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a precise academic annotator."},
            {"role": "user", "content": ANNOTATION_PROMPT + slide_text}
        ]
    )

    return response.choices[0].message.content


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    txt_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".txt")]

    for file_name in tqdm(txt_files, desc="Annotating slide decks"):
        file_path = os.path.join(INPUT_DIR, file_name)
        output_path = os.path.join(OUTPUT_DIR, file_name.replace(".txt", "_annotations.txt"))

        try:
            result = annotate_text_file(file_path)
            with open(output_path, "w") as f:
                f.write(result)
        except Exception as e:
            print(f"Error on {file_name}: {e}")

    print("All slide decks annotated!")


if __name__ == "__main__":
    main()

from preprocess.lang_processors.java_processor \
    import JavaProcessor

from preprocess.lang_processors.cpp_processor import CppProcessor
from preprocess.lang_processors.python_processor import PythonProcessor

java_code = r"""class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!"); 
    }
}"""

cpp_code = r"""
static void goodG2B1() {     char * data;     char dataBuffer[100] = "";     data = dataBuffer;     if(5!=5)     {         /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */         printLine("Benign, fixed string");     }     else     {         /* FIX: get the hostname from a string literal */         strcpy(data, "hostname");     }     /* POTENTIAL FLAW: set the hostname to data obtained from a potentially external source */     if (!SetComputerNameA(data))     {         printLine("Failure setting computer name");         exit(1);     } }
"""

java_processor = JavaProcessor()
cpp_processor = CppProcessor()
python_processor = PythonProcessor()

tokenized_java_code = java_processor.tokenize_code(java_code)
tokenized_cpp_code = cpp_processor.tokenize_code(cpp_code)


print(tokenized_cpp_code)

funs = ''
for token in tokenized_java_code:
    funs = funs + token + " "


print(funs)



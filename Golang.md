java complex but fast

python easy but slow interpreter

C slow for compliation time



to over come We use GO, Developed by Google in 2007.



**Features**                        inbuild support for garbage collections, strong nd statiscally typed, simplicity, FAST compile time.



**Where to use**



Web development (server-side)

Developing network-based programs

Developing cross-platform enterprise applications

Cloud-native development



**Why to use**



good memory management, works cross platform, fast runtime and  compiled time



**Structure**

package Declaration

import package

functions

variable

statements and expression

comments



**.go** Extension eg                 hello.go





step 1



\-**Declararion package**             package main



\-**package import**                  import ("fmt")



\-**function declaration**



func main(){

    fmt.Println("hello world")

}



**command**                           /\* \*/                                            //



**To run**                            go run filename.go



**To check version**                  go version



**To save .exe file** (.exe files are used because they're the standard way to distribute and run software on Windows.)            go build filename.go



**Variable Declaration**              var variablename type = value                     variablename := value



**multiple variable**                 var a, b, c, d int = 1, 3, 5, 7                   var a, b = 6, "Hello"                      c, d := 7, "World!"



**var**	                                                                       **:=**

Can be used inside and outside of functions	                               Can only be used inside functions

Variable declaration and value assignment can be done separately	       Variable declaration and value assignment cannot be done separately (must be done in the same line)





The **const** keyword declares the variable as "constant", which means that it is unchangeable and read-only.                      eg; const PI=3.14                typed, untyped constant

 Output Functions                  print()                                           printf("%v",i)                             println()

Verb	Description

**%v	Prints the value in the default format                                      fmt.Printf("%v\\n", txt)                    Hello World!**

**%#v	Prints the value in Go-syntax format                                        fmt.Printf("%#v\\n", txt)                   "Hello World!"**

**%T	Prints the type of the value                                                fmt.Printf("%T\\n", txt)                    string**

**%%	Prints the % sign
%b	Base 2                                                                      fmt.printf("%b\\n", i)                      1111**

**%d	Base 10                                                                     fmt.Printf("%d\\n", i)                      15**

**%+d	Base 10 and always show sign                                                fmt.Printf("%+d\\n", i)                     +15**

**%o	Base 8                                                                      fmt.Printf("%o\\n", i)                      17**

**%O	Base 8, with leading 0o                                                     fmt.Printf("%O\\n", i)                      0o17**

**%x	Base 16, lowercase                                                          fmt.Printf("%x\\n", i)                      f**

**%X	Base 16, uppercase                                                          fmt.Printf("%X\\n", i)                      F**

**%#x	Base 16, with leading 0x                                                    fmt.Printf("%#x\\n", i)                     0xf**

**%4d	Pad with spaces (width 4, right justified)                                  fmt.Printf("%4d\\n", i)                       15**

**%-4d	Pad with spaces (width 4, left justified)                                   fmt.Printf("%-4d\\n", i)                    15**

**%04d	Pad with zeroes (width 4                                                    fmt.Printf("%04d\\n", i)                    0015
%s	Prints the value as plain string                                            fmt.Printf("%s\\n", txt)                       Hello**

**%q	Prints the value as a double-quoted string                                  fmt.Printf("%q\\n", txt)                    "Hello"**

**%8s	Prints the value as plain string (width 8, right justified)                 fmt.Printf("%8s\\n", txt)                   Hello**

**%-8s	Prints the value as plain string (width 8, left justified)                  fmt.Printf("%-8s\\n", txt)                  Hello**

**%x	Prints the value as hex dump of byte values                                 fmt.Printf("%x\\n", txt)                    48656c6c6f**

 % x	Prints the value as hex dump with spaces                                    fmt.Printf("% x\\n", txt)                   48 65 6c 6c 6f

input getting

fmt.Scanln(\&a)

fmt.Println(a)



Default values            int 0                                                   String                                         Bool false

**Datatypes**                 int, uint- Signed, Unsigned(non-negative) 8,16,32,64      float 32,64                String          Bool          Complex 64,128



ru:='a'
fmt.Printf("%v %t",a,a)                           97 int32                returns ascii value and datatype



**Arrays**

With Var                  var array\_name = \[length]datatype{values} // here length is defined                  var array\_name = \[...]datatype{values} // here length is inferred

With :=                   array\_name := \[length]datatype{values} // here length is defined                     array\_name := \[...]datatype{values} // here length is inferred

**Eg:**

var arr1 = \[3]int{1,2,3}                    var arr1 = \[...]int{1,2,3}

arr2 := \[5]int{4,5,6,7,8}                   arr2 := \[...]int{4,5,6,7,8}



**Slices** are similar to arrays, but are more powerful and flexible. Like arrays, slices are also used to store multiple values of the same type in a single variable.

slice\_name := \[]datatype{values}            myslice := \[]int{1,2,3,}
**len()** function - returns the length of the slice (the number of elements in the slice)

**cap()** function - returns the capacity of the slice (the number of elements the slice can grow or shrink to)



**Create a Slice From an Array**



var arr1 = \[]int{1,2,3}

myans := arr1\[start:end]                                                           myslice := arr1\[2:4]

**Create a Slice With The make() Function**



slice\_name := make(\[]type, length, capacity)                                       myslice1 := make(\[]int, 5, 10)



You can append elements to the end of a slice using the append()function

slice\_name = append(slice\_name, element1, element2, ...)                           myslice1 = append(myslice1, 20, 21)

**ADD two r more slice**                      slice3 = append(slice1, slice2...)       myslice3 := append(myslice1, myslice2...)



The copy() function takes in two slices dest and src, and copies data from src to dest. It returns the number of elements copied.

copy(dest, src)                                                                    copy(numbersCopy, neededNumbers)

**Arithmetic**  Operators

* Addition	                                                        Adds together two values	                        x + y

**-	Subtraction                                                            	Subtracts one value from another	                x - y**

**\*	Multiplication                                                   	Multiplies two values	                                x \* y**

**/	Division	                                                        Divides one value by another                    	x / y**

**%	Modulus                                                               	Returns the division remainder	                        x % y**

**++	Increment                                                       	Increases the value of a variable by 1	                x++**

**--	Decrement                                                       	Decreases the value of a variable by 1	                x--**



Operator	                                                                Example	                                                Same As

=	                                                                        x = 5                                            	x = 5

+=	                                                                        x += 3	                                                x = x + 3

-=	                                                                        x -= 3	                                                x = x - 3

\*=	                                                                        x \*= 3                                           	x = x \* 3

/=	                                                                        x /= 3	                                                x = x / 3

%=	                                                                        x %= 3	                                                x = x % 3

\&=	                                                                        x \&= 3	                                                x = x \& 3

|=	                                                                        x |= 3	                                                x = x | 3

^=	                                                                        x ^= 3	                                                x = x ^ 3

>>=	                                                                        x >>= 3	                                                x = x >> 3	

<<=	                                                                        x <<= 3                                         	x = x << 3



Operator	                                                                Name	                                                Example

==	                                                                        Equal to	                                        x == y

!=	                                                                        Not equal	                                        x != y

>	                                                                        Greater than	                                        x > y	

<	                                                                        Less than                                       	x < y

>=	                                                                        Greater than or equal to	                        x >= y	

<=	                                                                        Less than or equal to	                                x <= y



Operator                                     	Name	                                            Description	                                                       Example

\&\& 	                                        Logical and	                                    Returns true if both statements are true	                       x < 5 \&\&  x < 10

|| 	                                        Logical or	                                    Returns true if one of the statements is true                      x < 5 || x < 4

!                                              	Logical not	                                    Reverse the result, returns false if the result is true	       !(x < 5 \&\& x < 10)

\& 	                                        Bitwise AND	                                    Sets each bit to 1 if both bits are 1	                       x \& y

|                                             	bitwise OR	                                    Sets each bit to 1 if one of two bits is 1	                       x | y

 ^	                                        bitwise XOR	                                    Sets each bit to 1 if only one of two bits is 1	               x ^ b

<<                                       	Zero fill left shift	                            Shift left by pushing zeros in from the right                      x << 2

>>	                              Signed right shift      Shift right by pushing copies of the leftmost bit in from the left, and let the rightmost bits fall off  x >> 2



**The Condition Statement**

if condition {

  // code to be executed if condition is true

}

**if..else statement**

if condition {

  // code to be executed if condition is true

} else {

  // code to be executed if condition is false

}

**elseif statement**

if condition1 {

   // code to be executed if condition1 is true

} else if condition2 {

   // code to be executed if condition1 is false and condition2 is true

} else {

   // code to be executed if condition1 and condition2 are both false

}

**Nested if**



if condition1 {

   // code to be executed if condition1 is true

  if condition2 {

     // code to be executed if both condition1 and condition2 are true

  }

}

**Switch**

switch expression {

case x:

   // code block

case y:

   // code block

case z:

...

default:

   // code block

}

The default keyword specifies some code to run if there is no case match:



 Example

func main() {

**day := 4**



**switch day {**

**case 1:**

     fmt.Println("Monday")

     fallthrough  \\\\eventhough it true ,it check next conditions too

**case 2:**

     fmt.Println("Tuesday")



**case 3:**

     fmt.Println("Wednesday")



**case 4:**

     fmt.Println("Thursday")



**case 5:**

     fmt.Println("Friday")



**case 6:**

     fmt.Println("Saturday")



**case 7:**

     fmt.Println("Sunday")

default:

     fmt.Println("Not a weekday")



 }

The Multi-case switch Statement



switch expression {

case x,y:

   // code block if expression is evaluated to x or y

case v,w:

   // code block if expression is evaluated to v or w

case z:

...

default:

   // code block if expression is not found in any cases

}



&nbsp;Goto Loop

Loop: 

     //statement 



     Goto Loop   (call again)



**For Statement**

for statement1; statement2; statement3 {                                                                    for i:=0; i < 5; i++ {

   // code to be executed for each iteration                                                                    fmt.Println(i)

}                                                                                                           }



The continue statement is used to skip one or more iterations in the loop. It then continues with the next iteration in the loop.

The break statement is used to break/terminate the loop execution.

The range keyword is used to more easily iterate through the elements of an array, slice or map. It returns both the index and the value.



for index, value := range array|slice|map {

   // code to be executed for each iteration

}

**Example**

package main

import ("fmt")



func main() {

  fruits := \[3]string{"apple", "orange", "banana"}

  for index, val := range fruits {

     fmt.Printf("%v\\t%v\\n", index, val)

  }

}



**Functions**

A function is a block of statements that can be used repeatedly in a program.

A function will not execute automatically when a page loads.

A function will be executed by a call to the function.



**Syntax**

**func FunctionName(param1 type, param2 type, param3 type) {**

**// code to be executed**

**}
Function calling**

FunctionName(val1,val2,val3)



**Function with return type**

func FunctionName(param1 type, param2 type) type {                                                            func myFunction(x int, y int) (result int) {

// code to be executed                                                                                              result = x + y

  return output                                                                                                     return  result

}                                                                                                             }



**Recursion Functions**

testcount() is a function that calls itself. We use the x variable as the data, which increments with 1 (x + 1) every time we recurse.

factorial\_recursion() is a function that calls itself. We use the x variable as the data, which decrements (-1) every time we recurse.



**Declare a Struct**

To declare a structure in Go, use the type and struct keywords

type struct\_name struct {                                                                                     type Person struct {

  member1 datatype;                                                                                                 name string

  member2 datatype;                                                                                                 age int

  member3 datatype;                                                                                                 job string

  ...                                                                                                           }

}



**Pointer**
ptr:= new(int)                                                ptr will be address

\*ptr = 90                                                     \*ptr will be value 90                            \&ptr will show address



**Access Struct Members**

To access any member of a structure, use the dot operator (.) between the structure variable name and the structure member:

package main

import ("fmt")



type Person struct {

  name string

  age int

  job string

  salary int

}



func main() {

  var pers1 Person

  var pers2 Person



  // Pers1 specification

  pers1.name = "Hege"

  pers1.age = 45

  pers1.job = "Teacher"

  pers1.salary = 6000



  // Pers2 specification

  pers2.name = "Cecilie"

  pers2.age = 24

  pers2.job = "Marketing"

  pers2.salary = 4500



  // Print Pers1 info by calling a function

  printPerson(pers1)



  // Print Pers2 info by calling a function

  printPerson(pers2)

}



func printPerson(pers Person) {

  fmt.Println("Name: ", pers.name)

  fmt.Println("Age: ", pers.age)

  fmt.Println("Job: ", pers.job)

  fmt.Println("Salary: ", pers.salary)

}



**Create Maps Using var and :=**



**var a = map\[KeyType]ValueType{key1:value1, key2:value2,...}**

 b := map\[KeyType]ValueType{key1:value1, key2:value2,...}

Example

package main

import ("fmt")



func main() {

  var a = map\[string]string{"brand": "Ford", "model": "Mustang", "year": "1964"}

  b := map\[string]int{"Oslo": 1, "Bergen": 2, "Trondheim": 3, "Stavanger": 4}



  fmt.Printf("a\\t%v\\n", a)

  fmt.Printf("b\\t%v\\n", b)

 }

Output
a	map\[brand:Ford model:Mustang year:1964]

b	map\[Bergen:2 Oslo:1 Stavanger:4 Trondheim:3]

**Create Maps Using the make() Function**

var a = make(map\[KeyType]ValueType)

b := make(map\[KeyType]ValueType)



**Example**
package main

import ("fmt")



func main() {

  var a = make(map\[string]string) // The map is empty now

  a\["brand"] = "Ford"

  a\["model"] = "Mustang"

  a\["year"] = "1964"

                                 // a is no longer empty

  b := make(map\[string]int)

  b\["Oslo"] = 1

  b\["Bergen"] = 2

  b\["Trondheim"] = 3

  b\["Stavanger"] = 4



  fmt.Printf("a\\t%v\\n", a)

  fmt.Printf("b\\t%v\\n", b)

}



**Output**
a   map\[brand:Ford model:Mustang year:1964]

b   map\[Bergen:2 Oslo:1 Stavanger:4 Trondheim:3]



**Empty map**

var a make(map\[KeyType]ValueType)

**Access Map Elements**

var a = make(map\[string]string)



  a\["brand"] = "Ford"

  a\["model"] = "Mustang"

  a\["year"] = "1964"



Removing elements is done using the delete() function

delete(map\_name, key)



func main() {

  a := map\[string]int{"one": 1, "two": 2, "three": 3, "four": 4}



  for k, v := range a {

    fmt.Printf("%v : %v, ", k, v)

  }

}

Result:



two : 2, three : 3, four : 4, one : 1,



**Goroutines**

A goroutine is a lightweight thread managed by the Go runtime. It allows your program to run  multiple functions concurrently  — not sequentially.

Think of it as a  function running in the background .



**Syntax** 
go functionName()

package main
import (
   "fmt"
   "time"
)
func printNumbers() {
   for i := 1; i <= 5; i++ {
       fmt.Println("Number:", i)

&nbsp;      time.Sleep(500 \\\* time.Millisecond)
   }

}



**Channels** 

A  channel  is a  communication mechanism  between goroutines. It allows one goroutine to  send data  and another to  receive data , safely and synchronously.

Think of it as a  pipe  through which goroutines send messages.



**Syntax** 
ch := make(chan DataType)


make() creates a channel.

<- is used to send or receive data.


Send \& Receive 


ch <- value     // Send value to channel
val := <-ch     // Receive value from channel

Example: Simple Channel 


package main
import "fmt"
func sendData(ch chan string) {
   ch <- "Hello from Goroutine!"
}
func main() {
  ch := make(chan string)
   go sendData(ch)     // goroutine sending data
   msg := <-ch         // main receives data
   fmt.Println(msg)
}

Hello from Goroutine!


**Buffered Channels** 



You can also make channels that hold multiple values before blocking.



Syntax 


ch := make(chan int, 3) // buffer of size 3

package main
import "fmt"
func main() {
   ch := make(chan int, 3)
   ch <- 10
   ch <- 20
   ch <- 30
   fmt.Println(<-ch)
   fmt.Println(<-ch)

&nbsp;  fmt.Println(<-ch)
}

Channels + Goroutines Example 

package main
import "fmt"
func add(a, b int, ch chan int) {

&nbsp;  result := a + b
   ch <- result
}
func main() {
   ch := make(chan int)
   go add(10, 20, ch)
   go add(5, 15, ch)
   x := <-ch
   y := <-ch
   fmt.Println("Results:", x, y)
}

Results: 30 20

**ERROR HANDLING \& EXCEPTIONS IN GO**


**Syntax**
value, err := someFunction()

if err != nil {

&nbsp;   // handle the error

}


**Example**
package main

import (

&nbsp;   "errors"

&nbsp;   "fmt"

)



func divide(a, b int) (int, error) {

&nbsp;   if b == 0 {

&nbsp;       return 0, errors.New("cannot divide by zero")

&nbsp;   }

&nbsp;   return a / b, nil

}



func main() {

&nbsp;   result, err := divide(10, 0)

&nbsp;   if err != nil {

&nbsp;       fmt.Println("Error:", err)

&nbsp;       return

&nbsp;   }

&nbsp;   fmt.Println("Result:", result)

}


**Custom Error Message**
package main

import (

&nbsp;   "fmt"

)



func readFile(filename string) error {

&nbsp;   return fmt.Errorf("failed to open file: %s", filename)

}



func main() {

&nbsp;   err := readFile("data.txt")

&nbsp;   if err != nil {

&nbsp;       fmt.Println("Error occurred:", err)

&nbsp;   }

}


**Using panic and recover**
Panic: Immediately stops normal execution of the current function. All deferred functions are executed. Used for unrecoverable errors (e.g., nil pointer, missing config).

recover: Used inside a defer block to regain control after a panic. Prevents the program from crashing completely.

package main

import "fmt"



func riskyOperation() {

&nbsp;   defer func() {

&nbsp;       if r := recover(); r != nil {

&nbsp;           fmt.Println("Recovered from panic:", r)

&nbsp;       }

&nbsp;   }()



&nbsp;   fmt.Println("Starting risky operation...")

&nbsp;   panic("something went wrong!") // simulate fatal error

&nbsp;   fmt.Println("This line will not execute")

}



func main() {

&nbsp;   riskyOperation()

&nbsp;   fmt.Println("Program continues normally after recovery.")

}

**FILE HANDLING IN GO**

file, err := os.Open("filename.txt")


package main

import (

&nbsp;   "bufio"

&nbsp;   "fmt"

&nbsp;   "os"

)



func main() {

&nbsp;   file, err := os.Open("example.txt")

&nbsp;   if err != nil {

&nbsp;       fmt.Println("Error:", err)

&nbsp;       return

&nbsp;   }

&nbsp;   defer file.Close()



&nbsp;   scanner := bufio.NewScanner(file)

&nbsp;   for scanner.Scan() {

&nbsp;       fmt.Println(scanner.Text())

&nbsp;   }



&nbsp;   if err := scanner.Err(); err != nil {

&nbsp;       fmt.Println("Error reading file:", err)

&nbsp;   }

}


package main

import (

&nbsp;   "fmt"

&nbsp;   "os"

)



func main() {

&nbsp;   file, err := os.OpenFile("output.txt", os.O\_APPEND|os.O\_WRONLY, 0644)

&nbsp;   if err != nil {

&nbsp;       fmt.Println("Error opening file:", err)

&nbsp;       return

&nbsp;   }

&nbsp;   defer file.Close()



&nbsp;   \_, err = file.WriteString("\\nAppended line using Go!")

&nbsp;   if err != nil {

&nbsp;       fmt.Println("Error writing to file:", err)

&nbsp;       return

&nbsp;   }



&nbsp;   fmt.Println("Data appended successfully!")

}


**Buildin Functions**

Function	                                                Description	                                        Example

len()                                                   	Returns length	                                        len(slice)

cap()                                                   	Returns capacity	                                cap(slice)

append()	                                                Adds elements to slice	                                append(s, x)

copy()                                                   	Copies slice elements	                                copy(dst, src)

make()                                                   	Allocates slice/map/channel                       	make(\[]int, 5)

new()                                                   	Allocates memory                                   	new(int)

delete()                                                	Removes map key	                                        delete(m, key)

close()                                                   	Closes channel	                                        close(ch)

panic()	                                                        Triggers runtime panic	                                panic("error")

recover()                                                 	Handles panic	                                        recover()

print() / println()	                                        Prints text (debug)	                                println("Hi")

complex()	                                                Makes complex number	                                complex(3,4)

real() / imag()	                                                Extract real/imag part	                                real(c)

Which **Subjects Are Go Relevant For?**



Backend Development:

Go is excellent for building server-side applications.



Cloud Computing:

Go is widely used in cloud infrastructure.



System Programming:

Go provides low-level system access.



Microservices:

Go excels at building microservices.



DevOps:

Go is popular for DevOps tooling.



Network Programming:

Go has strong networking capabilities.



Concurrent Programming:

Go makes concurrent programming simple.



 Go Interview Questions Basic

1\.  What is Go programming language, and why is it used?

Go is a modern programming language developed by Google. It is designed to be simple, efficient, and reliable. It is often used for building scalable and highly concurrent applications. It combines the ease of use of a high-level language with the performance of a low-level language. Go's syntax is easy to understand and its standard library provides a wide range of functionalities.

Additionally, Go has built-in support for concurrency, making it ideal for developing applications that require dealing with multiple tasks simultaneously. Overall, Go is used for developing fast, efficient, and robust software, especially in the field of web development and cloud computing.



**2.** How do you implement concurrency in Go?

In Go, concurrency is implemented using goroutines and channels. Goroutines are lightweight threads that can be created with the go keyword. They allow concurrent execution of functions.

Channels, on the other hand, are used for communication and synchronization between goroutines. They can be created using make and can be used to send and receive values.

To start a goroutine, simply prefix a function call with the go keyword. This will create a new goroutine that executes the function concurrently. Channels can be used to share data between goroutines, allowing them to communicate and synchronize.

By using goroutines and channels, Go provides a simple and efficient way to implement concurrency in programs.



**3.** How do you handle errors in Go?

In Go, errors are handled using the error type. When a function encounters an error, it can return an error value indicating the problem. The calling code can then check if the error is nil. If not, it handles the error appropriately.

Go provides a built-in panic function to handle exceptional situations. When a panic occurs, it stops the execution of the program and starts unwinding the stack, executing deferred functions along the way. To recover from a panic, you can use the recover function in a deferred function. This allows you to handle the panic gracefully and continue the program execution.



**4.** How do you implement interfaces in Go?

Interfaces are implemented implicitly in Go. This means that you don't need to explicitly declare that a type implements an interface. Instead, if a type satisfies all the methods defined in an interface, it is considered to implement that interface.

This is done by first defining the interface type using the type keyword followed by the interface name and the methods it should contain. The next step is creating a struct type or any existing type that has all the methods required by the interface. Go compiler will automatically recognize that type as implementing the interface.

Using interfaces can help you achieve greater flexibility and polymorphism in Go programs.



**5.** How do you optimize the performance of Go code?

These strategies can optimize Go code performance:

Minimize memory allocations: Avoid unnecessary allocations by reusing existing objects or using buffers.

Use goroutines and channels efficiently: Leverage the power of concurrent programming, but ensure proper synchronization to avoid race conditions.

Optimize loops and conditionals: Reduce the number of iterations by simplifying logic or using more efficient algorithms.

Profile your code: Use Go's built-in profiling tools to identify bottlenecks and hotspots in your code.



**6.** What is the role of the "init" function in Go?

The "init" function is a special function in Go that is used to initialize global variables or perform any other setup tasks needed by a package before it is used. The init function is called automatically when the package is first initialized. Its execution order within a package is not guaranteed.

Multiple init functions can be defined within a single package and even within a single file. This allows for modular and flexible initialization of package-level resources. Overall, the init function plays a crucial role in ensuring that packages are correctly initialized and ready to use when they are called.



**7.** What are dynamic and static types of declaration of a variable in Go?

The compiler must interpret the type of variable in a dynamic type variable declaration based on the value provided to it. The compiler does not consider it necessary for a variable to be typed statically.

Static type variable declaration assures the compiler that there is only one variable with the provided type and name, allowing the compiler to continue compiling without having all of the variable's details. A variable declaration only has meaning when the program is being compiled; the compiler requires genuine variable declaration when the program is being linked.



**8.** What is the syntax for declaring a variable in Go?

In Go, variables can be declared using the var keyword followed by the variable name, type, and optional initial value. For example:

var age int = 29

Go also allows short variable declaration using the := operator, which automatically infers the variable type based on the assigned value. For example:

age := 29

In this case, the type of the variable is inferred from the value assigned to it.



**9.** What are Golang packages?

This is a common Golang interview question. Go Packages (abbreviated pkg) are simply directories in the Go workspace that contain Go source files or other Go packages. Every piece of code created in the source files, from variables to functions, is then placed in a linked package. Every source file should be associated with a package.



**10.** What are the different types of data types in Go?

The various data types in Go are:

Numeric types: Integers, floating-point, and complex numbers.

Boolean types: Represents true or false values.

String types: Represents a sequence of characters.

Array types: Stores a fixed-size sequence of elements of the same type.

Slice types: Serves as a flexible and dynamic array.

Struct types: Defines a collection of fields, each with a name and a type.

Pointer types: Holds the memory address of a value.

Function types: Represents a function.



**11.** How do you create a constant in Go?

To create a constant in Go, you can use the const keyword, followed by the name of the constant and its value. The value must be a compile-time constant such as a string, number, or boolean. Here's an example:

const Pi = 3.14159

After defining a constant, you can use it in your code throughout the program. Note that constants cannot be reassigned or modified during the execution of the program.

Creating constants allows you to give meaningful names to important values that remain constant throughout your Go program.



**12.** What data types does Golang use?

This is a common golang interview question. Golang uses the following types:

Slice

Struct

Pointer

Function

Method

Boolean

Numeric

String

Array

Interface

Map

Channel



**13.** Distinguish unbuffered from buffered channels.

This is a popular Golang interview question. The sender will block on an unbuffered channel until the receiver receives data from the channel, and the receiver will block on the channel until the sender puts data into the channel.

The sender of the buffered channel will block when there is no empty slot on the channel, however, the receiver will block on the channel when it is empty, as opposed to the unbuffered equivalent.



**14.** Explain string literals.

A string literal is a character-concatenated string constant. Raw string literals and interpreted string literals are the two types of string literals. Raw string literals are enclosed in backticks (foo) and contain uninterpreted UTF-8 characters. Interpreted string literals are strings that are written within double quotes and can contain any character except newlines and unfinished double-quotes.



**15.** What is a Goroutine and how do you stop it?

A Goroutine is a function or procedure that runs concurrently with other Goroutines on a dedicated Goroutine thread. Goroutine threads are lighter than ordinary threads, and most Golang programs use thousands of goroutines at the same time.

A Goroutine can be stopped by passing it a signal channel. Because Goroutines can only respond to signals if they are taught to check, you must put checks at logical places, such as at the top of your for a loop.



**16.** What is the syntax for creating a function in Go?

To create a function in Go, you need to use the keyword func, followed by the function name, any parameter(s) enclosed in parentheses, and any return type(s) enclosed in parentheses. The function body is enclosed in curly braces {}.

Here is an example function that takes two integers as input and returns their sum:

We declare a function called add that takes two parameters, x and y, and returns their sum as an int.



**17.** How do you create a loop in Go?

The most commonly used loop is the for loop. It has three components: the initialization statement, the condition statement, and the post statement.

Here is an example of a for loop:

In this example, the loop will iterate 10 times. You can modify the i, condition, and post statement to customize the loop behavior.



**18.** What is the syntax for an if statement in Go?

The syntax for an if statement in Go is straightforward and similar to other programming languages. The if keyword is followed by a condition enclosed in parentheses, and the body of the statement is enclosed in curly braces.

For example,

if statement in go.webp

This code block compares variables a and b and prints a message depending on their values. The condition is evaluated, and if it's true, the code inside the curly braces is executed. If it's false, the program skips to the else statement.



**19.** What are some benefits of using Go?

This is an important Golang interview question. Go is an attempt to create a new, concurrent, garbage-collected language with quick compilation and the following advantages:

On a single machine, a big Go application can be compiled in a matter of seconds.

Go provides an architecture for software development that simplifies dependency analysis while avoiding much of the complexity associated with C-style programs, such as files and libraries.

Because there is no hierarchy in Go's type system, no work is wasted describing the relationships between types. Furthermore, while Go uses static types, the language strives to make types feel lighter weight than in traditional OO languages.

Go is fully garbage-collected and supports parallel execution and communication at a fundamental level.

Go's design presents a method for developing system software on multicore processors.



**20.** How do you create a pointer in Go?

You can use the \& symbol, followed by a variable to create a pointer in Go. This returns the memory address of the variable. For example, if you have a variable num of type int, you can create a pointer to num like this:

var num int = 42

var ptr \*int = \&num

Here, ptr is a pointer to num. You can use the \* symbol to access the value stored in the memory address pointed by a pointer. For instance, \*ptr will give you the value 42. Pointers are useful for efficient memory sharing and passing references between functions.



**21.** What is the syntax for creating a struct in Go?

You need to define a blueprint for the struct, which may consist of fields of different data types. The blueprint for the struct is defined using the 'type' keyword, followed by the name you want to give the struct.

You then use the 'struct' keyword, followed by braces ('{}') where you list the fields, each with a name and a data type separated by a comma.

For instance, the syntax for creating a struct named Person with the fields name, age, and job of string, integer, and string data types, respectively, would be:



**22.** How do you create an array in Go?

Creating an array in Go is simple. First, you need to declare the array by specifying its type and size. You can do this by using the following syntax:

var myArray \[size]datatype

Replace size and datatype with the size and data type you want to use for your array. After declaring the array, you can then initialize it by assigning values to each index. You can also access and modify elements of the array using their index number.

Arrays in Go have fixed sizes, meaning you cannot add or remove elements once they are declared.



**23.** How will you perform inheritance with Golang?

This is a trick golang interview question because Golang does not support classes, hence there is no inheritance.

However, you may use composition to imitate inheritance behavior by leveraging an existing struct object to establish the initial behavior of a new object. Once the new object is created, the functionality of the original struct can be enhanced.



**24.** How do you create a slice in Go?

You first need to define a variable of type slice using the make() function. The make() function takes two arguments: the first is the type of the slice you want to create (for example, \[]string for a slice of strings) and the second is the length of the slice. The length of the slice is not fixed and can be changed dynamically as elements are added or removed.

Here’s an example to create a slice of strings with a length of 5:

mySlice := make(\[]string, 5)

You can access and modify the elements in the slice using their index.



**25.** What is the difference between an array and a slice in Go?

In Go, an array is a fixed-length sequence of elements of the same type. Once an array is defined, the length cannot be changed. On the other hand, a slice is a dynamically-sized, flexible view of an underlying array. It is created with a variable length and can be resized.

Slices are typically used when you need to work with a portion of an array or when you want to pass a subset of an array to a function. Slices provide more flexibility and are widely used in Go for their convenience and efficiency in managing collections of data.



**26.** How do you create a map in Go?

You can use the make keyword, followed by the map keyword and the data types for the key and value. The syntax would be make(map\[keyType]valueType).

For example, to create a map of string keys and integer values, you would use make(map\[string]int). You can assign values to the map using the bracket notation such as mapName\[key] = value. To access values, simply use mapName\[key].

Remember, maps in Go are unordered collections of key-value pairs, making them useful for storing and retrieving data efficiently.



**27.** How do you iterate through a map in Go?

To iterate through a map in Go, you can use a for loop combined with the range keyword. For example:

In this loop, key represents the key of each key-value pair in the map, and value represents the corresponding value. You can perform any desired operation within the loop. The range keyword automatically iterates over the map and gives you access to its keys and values.



**28.** What is a goroutine in Go?

A goroutine is a lightweight thread of execution that enables concurrent programming. It is a function that can be run concurrently with other goroutines. It is managed by the Go runtime and has a very small footprint compared to threads in other programming languages.

Goroutines are more efficient in terms of memory usage and can be created and destroyed quickly. They can communicate with each other through channels, which provide a safe way to exchange data and synchronize their execution. This allows for efficient and scalable concurrent programming in Go.



**29.** What are the looping constructs in Go?

There is only one looping construct in Go: the for loop. The for loop is made up of three parts that are separated by semicolons:

Before the loop begins, the Init statement is run. It is frequently a variable declaration that is only accessible within the scope of the for a loop.

Before each iteration, the condition statement is evaluated as a Boolean to determine if the loop should continue.

At the end of each cycle, the post statement is executed.



**30.** What is a channel in Go?

In Go, a channel is a data structure that allows goroutines (concurrent functions) to communicate and synchronize with each other. It can be thought of as a conduit through which you can pass values between goroutines. A channel has a specific type that indicates the type of values that can be sent and received on it.

Channels can be used to implement synchronization between goroutines and data sharing. They provide a safe and efficient way to coordinate the flow of information, ensuring that goroutines can send and receive data in a controlled and synchronized manner.



**31.** How do you create a channel in Go?

You can use the built-in make function with the chan keyword to create a channel in Go. Here's an example:

ch := make(chan int)

In the above code, a channel called ch has been created that can transmit integers. This channel can be used to send and receive data between goroutines.

By default, channels are unbuffered, meaning that the sender blocks until the receiver is ready. You can also create buffered channels by providing a buffer capacity as a second argument to the make function.

Channels are a powerful synchronization mechanism in Go, allowing safe communication and coordination between concurrent processes.



**32.** How do you close a channel in Go?

The close() function is used to close a channel in Go. The function is used to indicate that no more values will be sent through the channel. Once a channel is closed, any subsequent attempt to send data through it will result in a runtime panic. However, receiving from a closed channel is still possible.

With the built-in v, ok := <-ch syntax, you can receive values from a closed channel. The ok boolean flag will be set to false if the channel is closed. It's important to note that closed channels should only be used for signaling and not for synchronization.



**33.** How do you range over a channel in Go?

To range over a channel in Go, a for loop with the range keyword can be used. This allows you to iterate over all the values sent on the channel until it is closed. When using range, the loop will continue until the channel is closed or until no more values are available.

Here is an example of how to range over a channel in Go:

The loop will print the values 0 to 4 as they are sent on the channel.

Wrapping up

The list of questions provided here, especially the senior Golang developer questions, can help you solve similar queries and generate new ones. Keep in mind that an interview will not solely comprise Golang interview questions. You may be assessed on your communication skills, your past work, your ability to think on your feet, etc. This allows recruiters to determine how well you perform in difficult situations and in a team.



**Link**: https://www.turing.com/interview-questions/golang


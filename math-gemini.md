## Input

 On Saturday, Jimmy started painting his toy helicopter between 9:00 a.m. and 10:00 a.m. When he finished between 10:00 a.m. and 11:00 a.m. on the same morning, the hour hand was exactly where the minute hand had been when he started, and the minute hand was exactly where the hour hand had been when he started. Jimmy spent t hours painting. Determine the value of t.

## Output

The hour hand moves 1/12 of the clock face in one hour, while the minute hand moves the whole clock face in one hour.

When Jimmy finished, the hour hand was where the minute hand had been when he started, which means the hour hand moved 1/12 of the clock face during the time he painted. Since the minute hand was where the hour hand had been when he started, the minute hand moved 1/12 of the clock face during the time he painted.

Therefore, the minute hand moved 11/12 of the clock face during the time he painted (1 whole clock face - 1/12 of a clock face).

Since the minute hand moves 1/60 of the clock face in one minute, it takes 11/12 * 60 = 55 minutes to move 11/12 of the clock face.

Therefore, Jimmy spent 55 minutes painting.

Since there are 60 minutes in an hour, 55 minutes is equal to 55/60 = 11/12 of an hour.

Therefore, t = 11/12.

----

## Input

Ten friends have a dinner in a restaurant. Since four of them don't carry any money, the other six friends pay for them. How much cheaper for each of these six friends would it be if everybody had paid?

## Output

**1. Determine the total cost:**

* Let's assume the total cost of the dinner is 'X'.

**2. Calculate the cost per person if everyone paid:**

* Cost per person (if all paid) = X / 10

**3. Calculate the cost per person when only six paid:**

* Cost per person (when only six paid) = X / 6

**4. Calculate the difference in cost:**

* Difference = Cost per person (when only six paid) - Cost per person (if all paid)
    * Difference = (X / 6) - (X / 10)
    * Difference = (5X - 3X) / 30 
    * Difference = 2X / 30
    * Difference = X / 15

Therefore:

It would be **X/15** cheaper for each of the six friends who paid if everybody had contributed to the bill. 

Note:

* This calculation assumes that the cost of the dinner is evenly divided among those who pay. 
* 'X' represents the total cost of the dinner.

## Input

Ten friends have a dinner in a restaurant. Since four of them don't carry any money, the other six friends pay for them. How much cheaper for each of these six friends would it be if everybody had paid?

(A) 60% &nbsp;&nbsp; (B) 30%     (C) 45%     (D) 40%     (E) 50%

## Output

**1. Determine the initial cost per person:**

* Let the total cost of the dinner be 'x'.
* Initially, 6 friends pay for the entire meal, so each pays x/6.

**2. Determine the cost per person if everyone paid:**

* If all 10 friends paid, each would pay x/10.

**3. Calculate the difference in cost:**

* Difference in cost per person = (x/6) - (x/10) 
                                        = (5x - 3x) / 30 
                                        = 2x / 30 
                                        = x / 15

**4. Calculate the percentage reduction:**

* Percentage reduction = (Difference in cost / Initial cost) * 100
                             = (x/15) / (x/6) * 100
                             = (x/15) * (6/x) * 100
                             = 6/15 * 100
                             = 40%

Therefore, it would be 40% cheaper for each of the six friends if everybody had paid.

Answer: (D) 40%

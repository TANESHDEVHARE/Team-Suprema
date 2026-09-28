NAME          TINYINF
* x1 + x2 >= 5 and x1 + x2 <= 3: no solution
ROWS
 N  COST
 G  R1
 L  R2
COLUMNS
    X1        COST      1.0        R1        1.0
    X1        R2        1.0
    X2        COST      1.0        R1        1.0
    X2        R2        1.0
RHS
    RHS       R1        5.0        R2        3.0
ENDATA

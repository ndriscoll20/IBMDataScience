SELECT * FROM JOBS;
SELECT * FROM EMPLOYEES;
-- Exercise 1 - Sub-Queries
--Retrieve Employees records that correspond to jobs in JOBS TABLE
SELECT * FROM EMPLOYEES WHERE JOB_ID IN (SELECT JOB_IDENT from JOBS);

--Retrieve only list of employees whose JOB_TITLE is Jr. Designer
SELECT * FROM EMPLOYEES WHERE JOB_ID IN (SELECT JOB_IDENT FROM JOBS WHERE JOB_TITLE='Jr. Designer');

--Retrieve JOB information and who earn more than $70,000
SELECT * FROM JOBS WHERE JOB_IDENT IN (SELECT JOB_ID from EMPLOYEES WHERE SALARY > 70000);

-- Retrieve Job information and whose birth year is after 1976
SELECT * FROM JOBS WHERE JOB_IDENT IN (SELECT JOB_ID FROM EMPLOYEES WHERE YEAR(B_DATE) > 1976);

-- Exercise 2 - Accessing Multiple Tables with Implicit Joins
-- Cross Join between Employees and JOBS
SELECT * FROM EMPLOYEES, JOBS;
-- Retrieve only Employees that coresponse to jobs in Jobs table
SELECT * FROM EMPLOYEES, JOBS WHERE EMPLOYEES.JOB_ID = JOBS.JOB_IDENT;
-- Use Aliases
SELECT * FROM EMPLOYEES E, JOBS J WHERE E.JOB_ID = J.JOB_IDENT;
-- Retrieve only employee id name and title
SELECT E.EMP_ID, E.F_NAME, E.L_NAME, J.JOB_TITLE FROM EMPLOYEES E, JOBS J WHERE E.JOB_ID = J.JOB_IDENT;
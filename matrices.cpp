#include <iostream>
#include <vector>
#include "matrix.h"


// Using a template ensures the functions can be used with matrices containing
// integers, floats, doubles, or long doubles.

// Matrices in this file are almost always passed to functions by const
// reference to reduce memory use and save copying time.

// These functions are designed to be used with any user input, and therefore
// have data validation to ensure, for example, that the input matrices have
// the correct dimensions.

template <typename T>
struct LU_output{
    /**
     * struct LU_output
     * Stores the output from LU decomposition.
     *
     * Using a structure allows functions in C++ to return more than one object.
     *
     * Members:
     *   matrix_L: The lower triangular matrix from LU decomposition.
     *   matrix_U: The upper triangular matrix from LU decomposition.
     */
    Matrix<T> matrix_L;
    Matrix<T> matrix_U;
};

template <typename T>
LU_output<T> crout (const Matrix<T> & matrix_A) {
    /**
     * function crout
     * Uses Crout's method to carry out LU decomposition of an N*N matrix.
     *
     * This algorithm won't work as expected for a matrix of ints due to integer division.
     *
     * Parameters:
     *   Matrix<T> matrix_A: An N*N matrix to decompose. Works better if is type Matrix<double> or Matrix<float>.
     *
     * Returns:
     *   Matrix<T>: An N*N matrix containing all the elements of U and the non-diagonal elements of L.
     *
     * Errors:
     *   Throws an error if the matrix is not an N*N matrix.
     */

    unsigned rows = matrix_A.get_rows();
    unsigned cols = matrix_A.get_cols();

    // Data validation.
    try {
        // Check that matrix_A is square.
        if (rows != cols) {
            // Throw an error.
            throw 1;
        }
    } catch (int e) {
        // Print an error statenent.
        std::cout << "crout: The input matrix must be square." << std::endl;
        // Exit the program and report the failure.
        exit(EXIT_FAILURE);
    }

    // Initialise the L and U matrices.
    Matrix<T> matrix_U(rows, rows);
    Matrix<T> matrix_L(rows, rows);

    // Set the diagonal of L to 1.
    for (unsigned i = 0; i < rows; ++i) {
        matrix_L[i][i] = 1;
    }

    // Uses the algorithm from Computational Physics, Lecture 3 - Matrix
    // Methods, Slide 25 [P. Scott, 2017]

    // Iterate over each column in the matrix.
    for (unsigned j = 0; j < cols; ++j) {
        // Starting at the top, iterate over each entry in the column until
        // reaching the diagonal.
        for (unsigned i = 0; i <= j; ++i) {
            T total = 0;
            for (unsigned k = 0; k < i; ++k) {
                total += matrix_L[i][k] * matrix_U[k][j];
            }
            // Set the U matrix.
            matrix_U[i][j] = matrix_A[i][j] - total;
        }
        // Starting at the diagonal, iterate over each entry in the column until
        // reaching the bottom.
        for (unsigned i = j; i < rows; ++i) {
            T total = 0;
            for (unsigned k = 0; k < j; ++k) {
                total += matrix_L[i][k] * matrix_U[k][j];
            }
            // Set the L matrix.
            matrix_L[i][j] = (matrix_A[i][j] - total) / matrix_U[j][j];
        }
    }

    // Initialise the output structure.
    LU_output<T> output;

    // Set the values in the output structure.
    output.matrix_L = matrix_L;
    output.matrix_U = matrix_U;

    // Return the structure.
    return output;
}

template <typename T>
Matrix<T> combine_LU_output (const LU_output<T> & LU_structure) {
    /**
     * function combine_LU_output
     * Takes the output of LU decomposition and combines into a single matrix containing all the elements of U, and the
     * non-diagonal elements of L.
     *
     * Parameters:
     *   LU_output LU_structure<T>: The output structure from and LU decomposition algorithm.
     *
     * Returns:
     *   Matrix<T>: The combined matrix.
     *
     * Errors:
     *   Throws an error if LU_structure.matrix_L and output.matrix_U are different sizes.
     *   Throws an error if LU_structure.matrix_L is not square.
     */

    unsigned rows = LU_structure.matrix_L.get_rows();
    unsigned cols = LU_structure.matrix_L.get_cols();

    // Data validation.
    try {
        // Check that the matrices are the same size.
        if ((LU_structure.matrix_U.get_rows() != rows) && (LU_structure.matrix_U.get_cols() != cols)) {
            throw 1;
        }
        // Check the matrices are square.
        if (rows != cols) {
            throw 2;
        }
    } catch (int e) {
        // Print the relevant error statenent.
        if (e == 1) {
            std::cout << "combine_LU_output: The input matrices must be the same size." << std::endl;
        }
        if (e == 2) {
            std::cout << "combine_LU_output: The matrices must be square." << std::endl;
        }
        // Exit the program and report the failure.
        exit(EXIT_FAILURE);
    }

    // Initialise output matrix to contain the same entries as the U matrix.
    Matrix<T> output = LU_structure.matrix_U;

    // Iterate over each row in the matrix.
    for (unsigned i = 0; i < rows; ++i) {
        // Starting at the edge, iterate over each entry in the row until
        // reaching the diagonal.
        for (unsigned j = 0; j < i; ++j) {
            // Copy the entry of the L matrix at the current index into the
            // output matrix.
            output[i][j] = LU_structure.matrix_L[i][j];
        }
    }

    // Return the combined matrix.
    return output;

}

template <typename T>
bool check_lower_triangular(const Matrix<T> & matrix) {
    /**
     * function check_lower_triangular
     * Checks if a matrix is lower triangular.
     *
     * Parameters:
     *   Matrix<T> matrix: A Matrix object.
     *
     * Returns:
     *   bool: True iff matrix is lower triangular.
     */

    // Assume the matrix is lower triangular.
    bool lower_triangular = true;

    // Loop through the rows of the matrix.
    for (unsigned i = 0; i < matrix.get_rows(); ++i) {
        // Loop through items in the current row to the right of the diagonal.
        for (unsigned j = i + 1; j < matrix.get_cols(); ++j) {
            // If any of the matrix elements are non-zero, the matrix is not
            // lower triangular.
            if (matrix[i][j] != 0) {
                lower_triangular = false;
            }
        }
    }

    // Return the boolean.
    return lower_triangular;
}

template <typename T>
bool check_upper_triangular(const Matrix<T> & matrix) {
    /**
     * function check_upper_triangular
     * Checks if a matrix is upper triangular.
     *
     * Parameters:
     *   Matrix<T> matrix: A Matrix object.
     *
     * Returns:
     *   bool: True iff matrix is upper triangular.
     */

    // Assume the matrix is lower triangular.
    bool upper_triangular = true;

    // Loop through the rows of the matrix.
    for (unsigned i = 0; i < matrix.get_rows(); ++i) {
        // Loop through items in the current row to the right of the diagonal.
        for (unsigned j = 0; j < i; ++j) {
            // If any of the matrix elements are non-zero, the matrix is not
            // upper triangular.
            if (matrix[i][j] != 0) {
                upper_triangular = false;
            }
        }
    }

    // Return the boolean.
    return upper_triangular;
}

template <typename T>
bool check_triangular(Matrix<T> matrix) {
    /**
     * function check_triangular
     * Checks if a matrix is triangular.
     *
     * Parameters:
     *   Matrix<T> matrix: A Matrix object.
     *
     * Returns:
     *   bool: True iff matrix is triangular.
     */

    unsigned rows = matrix.get_rows();
    unsigned cols = matrix.get_cols();

    // Check that the matrix is square.
    if (cols != rows) {
        return false;
    }

    // Check that the matrix is triangular.
    bool lower_triangular = check_lower_triangular(matrix);
    bool upper_triangular = check_upper_triangular(matrix);

    // Return true if the matrix is either lower or upper triangular.
    return (lower_triangular || upper_triangular);
}

template <typename T>
T triangular_determinant(const Matrix<T> & triangular){
    /**
     * function triangular_determinant
     * Calculates the determinant of a triangular matrix.
     *
     * Uses the result that the determinant of a triangular matrix is the product of the diagonal entries.
     *
     * Parameters:
     *   Matrix<T> triangular: A triangular matrix with entries of type T.
     *
     * Returns:
     *   T determinant: The determinant of the matrix. Returned as the same type T as the matrix entries.
     *
     * Errors:
     *   Throws an error if the matrix is not triangular.
     */

    unsigned rows = triangular.get_rows();
    unsigned cols = triangular.get_cols();

    // Data validation.
    try {
        // Check that the matrix is square.
        if (cols != rows) {
            throw 1;
        }
        // Check that the matrix is triangular.
        if (!check_triangular(triangular)) {
            throw 2;
        }
    } catch (int e) {
        // Print the relevant error statenent.
        if (e == 1) {
            std::cout << "triangular_determinant: The input matrix is not square, so cannot be triangular." << std::endl;
        }
        if (e == 2) {
            std::cout << "triangular_determinant: The input matrix must be triangular." << std::endl;
        }
        // Exit the program and report the failure.
        exit(EXIT_FAILURE);
    }

    // Initialise the determinant to 1.
    T determinant = 1;

    // Loop through the rows of the matrix.
    for (unsigned i = 0; i < rows; ++i) {
        // Multiply the determinant by the element of the row on the diagonal.
        determinant *= triangular[i][i];
    }

    // Return the value of the determinant.
    return determinant;
}

template <typename T>
Matrix<T> linsolve_LU(const Matrix<T> & matrix_L, const Matrix<T> & matrix_U, const Matrix<T> & vector_b){
    /**
     * function linsolve_LU
     * Solves the equation Ax = b using the LU decomposition of matrix A.
     *
     * Parameters:
     *   Matrix<T> matrix_L: A lower triangular matrix with entries of type T.
     *   Matrix<T> matrix_U: An upper triangular matrix with entries of the same type T.
     *   Matrix<T> vector_B: A matrix with only one column, with entries of the same type T.
     *
     * Returns:
     *   Matrix<T>: A matrix x of type T that solves the equqtion LUx = b.
     *
     * Errors:
     *   Throws an error if the matrices or vector are inconsistent sizes.
     *   Throws an error if the matrices are not upper or lower triangular.
     */

    unsigned rows = matrix_L.get_rows();
    unsigned cols = matrix_L.get_cols();

    // Data validation.
    try {
        // Check that the matrices are the same size.
        if ((matrix_U.get_rows() != rows) && (matrix_U.get_cols() != cols)) {
            throw 1;
        }
        // Check the matrices are square.
        if (rows != cols) {
            throw 2;
        }
        // Check that matrix_L is lower triangular.
        if (!check_lower_triangular(matrix_L)) {
            throw 3;
        }
        // Check that matrix_U is upper triangular.
        if (!check_upper_triangular(matrix_U)) {
            throw 4;
        }
        // Check that vector_b has only one column.
        if (vector_b.get_cols() != 1) {
            throw 5;
        }
        // Check that vector_b has the correct number of rows.
        if (vector_b.get_rows() != cols) {
            throw 6;
        }
    } catch (int e) {
        // Print the relevant error statenent.
        switch(e) {
            case 1: std::cout << "linsolve_LU: matrix_L and matrix_U must be the same size." << std::endl;
                    break;
            case 2: std::cout << "linsolve_LU: matrix_L and matrix_U must be square." << std::endl;
                    break;
            case 3: std::cout << "linsolve_LU: matrix_L must be lower triangular." << std::endl;
                    break;
            case 4: std::cout << "linsolve_LU: matrix_U must be upper triangular" << std::endl;
                    break;
            case 5: std::cout << "linsolve_LU: vector_b must be a have only 1 column." << std::endl;
                    break;
            case 6: std::cout << "linsolve_LU: vector_b must have the same number of rows as matrix_L and matrix_U." << std::endl;
                    break;
        }
        // Exit the program and report the failure.
        exit(EXIT_FAILURE);
    }

    // Initialise a vector y, to solve Ly = b.
    // A std::vector is used because vector_y cannot be accessed outside this
    // function.
    std::vector<T> vector_y(rows);

    // Calculate y_0.
    vector_y[0] = vector_b[0][0]/matrix_L[0][0];

    // Loop through the other entries in y and calculate their values.
    for (unsigned i = 1; i < rows; ++i) {
        // Implement the forward substitution calculation from Computational
        // Physics, Lecture 3 - Matrix Methods, Slide 23 [P. Scott, 2017]
        T total = 0;
        for (unsigned j = 0; j < i; ++j) {
            total += matrix_L[i][j]* vector_y[j];
        }
        // Store the value to vector_y.
        vector_y[i] = (vector_b[i][0] - total)/matrix_L[i][i];
    }

    // Initialise the vector x, to solve LUx = b.
    Matrix<T> vector_x(1, rows);

    // Calculate the last entry in vector_x.
    vector_x[rows - 1][0] = vector_y[rows - 1] / matrix_U[rows - 1][rows - 1];

    // Loop through the other entries in x and calculate their values.
    // Using signed ints despite calculation with a std::vector size (rows) since
    // decrementing unsigned ints can cause overflow errors.
    for (int i = rows - 2; i >= 0; --i){
        // Implement the backward substitution algorithm from Computational
        // Physics, Lecture 3 - Matrix Methods, Slide 23 [P. Scott, 2017]
        T total = 0;
        for (unsigned j = i + 1; j < rows; ++j) {
            total += matrix_U[i][j] * vector_x[j][0];
        }
        // Store the value to vector_x.
        vector_x[i][0] = (vector_y[i] - total)/matrix_U[i][i];
    }

    return vector_x;
}

template<typename T>
Matrix<T> find_inverse (const Matrix<T> & matrix_L, const Matrix<T> & matrix_U) {
    /**
     * function find_inverse
     * Find the inverse of a matrix from its LU decomposition.
     *
     * Parameters:
     *   Matrix<T> matrix_L: A lower triangular matrix with entries of type T.
     *   Matrix<T> matrix_U: An upper triangular matrix with entries of the same type T.
     *
     * Returns:
     *   Matrix<T>: The inverse of the matrix given by L*U.
     */

    unsigned rows = matrix_L.get_rows();
    unsigned cols = matrix_L.get_cols();

    // Data validation.
    try {
        // Check that the matrices are the same size.
        if ((matrix_U.get_rows() != rows) && (matrix_U.get_cols() != cols)) {
            throw 1;
        }
        // Check the matrices are square.
        if (rows != cols) {
            throw 2;
        }
        // Check that matrix_L is lower triangular.
        if (!check_lower_triangular(matrix_L)) {
            throw 3;
        }
        // Check that matrix_U is upper triangular.
        if (!check_upper_triangular(matrix_U)) {
            throw 4;
        }
    } catch (int e) {
        // Print the relevant error statenent.
        switch(e) {
            case 1: std::cout << "linsolve_LU: matrix_L and matrix_U must be the same size." << std::endl;
                    break;
            case 2: std::cout << "linsolve_LU: matrix_L and matrix_U must be square." << std::endl;
                    break;
            case 3: std::cout << "linsolve_LU: matrix_L must be lower triangular." << std::endl;
                    break;
            case 4: std::cout << "linsolve_LU: matrix_U must be upper triangular" << std::endl;
                    break;
        }
        // Exit the program and report the failure.
        exit(EXIT_FAILURE);
    }

    // Initialise the inverse matrix.
    Matrix<T> inverse(rows, rows);

    // Loop through the rows of the inverse matrix.
    for (unsigned i = 0; i < rows; ++i) {

        // Find the ith cartesian basis vector.
        Matrix<T> unit_vector(1, rows);
        unit_vector[i][0] = 1;
        // Use the basis vector to solve for a column of the inverse matrix.
        Matrix<T> vector_x = linsolve_LU(matrix_L, matrix_U, unit_vector);
        // Loop through the solution vector.
        for (unsigned j = 0; j < rows; ++j) {
            // Store the values from the solution to the inverse matrix.
            inverse[j][i] = vector_x[j][0];
       }
    }

    return inverse;

}


int main(int argc, char ** argv) {

    // Create each row in the matrix.
    std::vector<double> row1 = {2.0, 1.0, 0.0, 0.0, 0.0};
    std::vector<double> row2 = {3.0, 8.0, 4.0, 0.0, 0.};
    std::vector<double> row3 = {0.0, 9.0, 20.0, 10.0, 0.0};
    std::vector<double> row4 = {0.0, 0.0, 22.0, 51.0, -25.0};
    std::vector<double> row5 = {0.0, 0.0, 0.0, -55.0, 60.0};

    // Create the vector of vectors.
    std::vector<std::vector <double> > matrix_A_arr = {row1, row2, row3, row4, row5};

    // Initialise the vector of vectors as type Matrix.
    Matrix<double> matrix_A(matrix_A_arr);

    // Use Crout's algorithm
    LU_output<double> LU = crout(matrix_A);

    // Print the matrices.
    std::cout << "Lower triangular matrix:" << std::endl;
    LU.matrix_L.print();
    std::cout << "Upper triangular matrix:" << std::endl;
    LU.matrix_U.print();

    // Create the combined matrix as requested and print it.
    Matrix<double> combined_matrix = combine_LU_output(LU);
    std::cout << "Combined matrix:" << std::endl;
    combined_matrix.print();

    // Calculate the determinants of the upper triangular matrix.
    double det_U = triangular_determinant(LU.matrix_U);

    // Calculate the determinant of the original matrix and print it.
    // The determinant of the lower triangular matrix is one by definition.
    double det_A = det_U;
    std::cout << "The determinent of A is " << det_A << '.' << std::endl;

    // Create the vector b to solve the equation Ax = b.
    Matrix<double> vector_b({{2}, {5}, {-4}, {8}, {9}});

    // Solve the equation Ax = b using the triangular matrices.
    Matrix<double> vector_x = linsolve_LU(LU.matrix_L, LU.matrix_U, vector_b);

    // Print vector x.
    std::cout << "The solution (x) to Ax = b is:" << std::endl;
    vector_x.print();

    // Double check the answer by re-calculating vector_b.
    Matrix<double> vector_b_calc = matrix_A * vector_x;
    std::cout << "The vector (b) in Ax = b, recalculated from x, is:" << std::endl;
    vector_b_calc.print();

    // Find the inverse matrix to A.
    Matrix<double> matrix_A_inv = find_inverse(LU.matrix_L,LU. matrix_U);
    std::cout << "The inverse of A is:" << std::endl;
    matrix_A_inv.print();

    // Double check the answer by calculating the identity matrix.
    Matrix<double> identity = matrix_A * matrix_A_inv;
    // Remove floating point errors.
    for (unsigned i = 0; i < identity.get_rows(); ++i) {
        for (unsigned j = 0; j < identity.get_rows(); ++j) {
            if (identity[i][j] < 1e-14) {
                identity[i][j] = 0;
            }
        }
    }
    // Print the idenity matrix.
    std::cout << "The identity matrix, calculated using the inverse, is:" << std::endl;
    identity.print();

}

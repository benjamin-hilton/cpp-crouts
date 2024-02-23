#ifndef MATRIX_H
#define MATRIX_H

// Define MATRIX_H at compile time to ensure the module is not loaded twice.
// Only load the module if MATRIX_H has not previously been defined.

// This file does not define any functions outside of the Matrix class, ensuring
// that conflicts cannot occur when it is included in a source file.

#include <iostream>
#include <vector>
#include <algorithm>

// Using a template allows the existence of matrices with integer, float, double
// or long double elements.

template <typename T>
class Matrix {
    /**
     * class Matrix
     * A class that implements matrix operations.
     *
     * The class is designed to work with any type T. To initialise the class, use Matrix <T>, e.g. Matrix <double>.
     *
     * Properties:
     *   Private:
     *     rows: The number of rows in the matrix.
     *     cols: The number of columns in the matrix.
     *     matrix: A vector of vectors storing the data.
     *
     * Methods:
     *   Public:
     *     Matrix: Class constructor.
     *     get_rows: Returns the number of rows in the matrix.
     *     get_cols: Returns the number of columns in the matrix
     *     print: Displays the matrix in the console.
     *     operator[]: Returns a row of the matrix given an index.
     *     operator+: Adds two matrices.
     *     operator+=: Adds to a matrix.
     *     operator-: Subtracts two matrices.
     *     operator-=: Subtracts from a matrix.
     *     operator*: Multiplies two matrices or multiplies a matrix by a scalar.
     *     operator*=: Multiplies and updates a matrix.
     *     operator/: Divides a matrix by a scalar.
     *     operator/=: Divides and updates a matrix.
     */
private:
    unsigned rows; /** The number of rows in the matrix. */
    unsigned cols; /** The number of columns in the matrix. */
    std::vector< std::vector<T> > matrix; /** A vector of vectors storing the data */
public:
    explicit Matrix (const std::vector< std::vector<T> > input_vector) {
        /**
         * function Matrix
         * Matrix class constructor with only 1 input.
         *
         * Parameters:
         *   std::vector< std::vector<T> > input_vector: A vector of vectors of type T used as the data for the matrix.
         *
         * Returns:
         *   Matrix <T>: A Matrix object containing the data.
         *
         * Errors:
         *   Throws an error if any of the vectors in input_vector is a different size to the first vector.
         */


        unsigned column_no = input_vector[0].size();

        // Data validation.
        try {
            // Loop through each vector in input_vector.
            for (const std::vector<T> & row : input_vector){
                // Check that the current vector is the same size as the first vector.
                if (row.size() != column_no) {
                    // Throw an error.
                    throw 1;
                }
            }
        }
        catch (int e) {
            // Print an error statement.
            std::cout << "Matrix constructor: Inconsistent number of columns" << std::endl;
            // Exit the program and report the failure.
            exit(EXIT_FAILURE);
        }

        // Initialise the class properties.
        rows = input_vector.size();
        cols = column_no;
        matrix = input_vector;
    }

    Matrix (const unsigned x_size, const unsigned y_size) {
        /**
         * function Matrix
         * Matrix constructor for an empty matrix.
         *
         * Parameters:
         *   const unsigned x_size: The number of columns of the new matrix.
         *   const unsigned y_size: The number of rows of the new matrix.
         *
         * Returns:
         *   Matrix <T>: An empty matrix with the desired number of rows and columsn
         */

        // Create an empty matrix.
        matrix = std::vector< std::vector<T> > (y_size, std::vector<T>(x_size));

        // Initialise the other class properties.
        rows = y_size;
        cols = x_size;
    }

    Matrix () {
        /**
         * function Matrix
         * Matrix constructor for an empty matrix of undetermined size.
         *
         * Returns:
         *   Matrix<T>: A empty matrix with no size.
         */

        // Create an empty matrix with no size.
        matrix = std::vector< std::vector<T> > (0, std::vector<T>(0));

        // Initialise the other class properties.
        rows = 0;
        cols = 0;
    };

    std::vector<T>& operator[] (const int index) {
        /**
         * function operator[]
         * Returns a row of the matrix given an index.
         *
         * Overrides the [] operator so that the matrix vector of vectors does not need to be accessed directly. A
         * second [] operator then be used to access the desired matrix entry. This can be used to change values in the
         * matrix.
         *
         * Parameters:
         *   const int index: The desired row index.
         *
         * Returns:
         *   std::vector<T>&: A reference to the desired vector.
         */
        return matrix[index];
    }

    const std::vector<T>& operator[] (const int index) const {
        /**
         * function operator[]
         * Returns a row of a constant matrix given an index.
         *
         * Overrides the [] operator for const Matrix objects so that the matrix vector of vectors does not need to be
         * accessed directly. A second [] operator then be used to access the desired matrix entry. This CANNOT be used
         * to change values in the matrix.
         *
         * Parameters:
         *   const int index: The desired row index.
         *
         * Returns:
         *   const std::vector<T>&: A reference to the desired vector.
         */
        return matrix[index];
    }

    unsigned get_rows(){
        /**
         * function get_rows
         * Returns the number of rows in the matrix.
         *
         * Returns:
         *   unsigned: The value of matrix.rows.
         */
        return rows;
    }

    unsigned get_cols(){
        /**
         * function get_cols
         * Returns the number of columns in the matrix.
         *
         * Returns:
         *   unsigned: The value of matrix.cols.
         */

        return cols;
    }

    unsigned get_rows() const {
        /**
         * function get_rows
         * Returns the number of rows in the matrix.
         *
         * Returns:
         *   unsigned: The value of matrix.rows.
         */
        return rows;
    }

    unsigned get_cols() const {
        /**
         * function get_cols
         * Returns the number of columns in the matrix.
         *
         * Returns:
         *   unsigned: The value of matrix.cols.
         */

        return cols;
    }

    // Declaring functions that are defined below.

    void print();
    Matrix<T> operator+ (const Matrix<T> & matrix2);
    Matrix<T>& operator+= (const Matrix<T> & matrix2);
    Matrix<T> operator- (const Matrix<T> & matrix2);
    Matrix<T>& operator-= (const Matrix<T> & matrix2);
    Matrix<T> operator* (const Matrix<T> & matrix2);
    Matrix<T>& operator*= (const Matrix<T> & matrix2);
    Matrix<T> operator* (T value);
    Matrix<T>& operator*= (T value);
    Matrix<T> operator/ (T value);
    Matrix<T>& operator/= (T value);

};

template <typename T>
void Matrix<T>::print () {
    /**
     * function print
     * Displays the matrix in the console. Member function of Matrix.
     */

    // Print a '[' and a newline character to start the output.
    std::cout << '[' << '\n';

    // Loop through each row of the matrix.
    for (unsigned i = 0; i < rows; ++i){
        // Print a '[' to start the row.
        std::cout << '[';
        // Loop through each element in the row.
        for (unsigned j = 0; j < cols; ++j){
            // Print the current element.
            std::cout << ' ' << matrix[i][j] << ' ';
        }
        // Print a ']' and a newline character to end the row.
        std::cout << ']' << '\n';
    }

    // Print a ']' to end the output.
    std::cout << ']' << std::endl;
}


template <typename T>
Matrix<T> Matrix<T>::operator+ (const Matrix<T> & matrix2){
    /**
     * function operator+
     * Adds two matrices.
     *
     * Overrides the + operator for two Matrix objects of the same type T.
     *
     * Parameters:
     *   const Matrix<T> matrix2: A second matrix to add to the first, of the same type T.
     *
     * Returns:
     *   Matrix<T>: The sum of the two matrices.
     *
     * Errors:
     *   Throws an error if the matrices are different shapes.
     */

    // Data validation.
    try {
        // Check that the matrices are the same shape.
        if (!(rows == matrix2.rows && cols == matrix2.cols)) {
            // Throw an error.
            throw 1;
        }
    }
    catch (int e) {
        // Print an error statenent.
        std::cout << "Matrix addition: matrices different shapes" << std::endl;
        // Exit the program and report the failure.
        exit(EXIT_FAILURE);
    }

    // Create output matrix.
    Matrix<T> output(cols, rows);

    // Loop through the rows of the matrix.
    for (unsigned i = 0; i < rows; ++i){
        // Loop through each item in the row.
        for (unsigned j = 0; j < cols; ++j){
            // Add the values of the matrices at the current index.
            output[i][j] = matrix[i][j] + matrix2[i][j];
        }
    }

    // Return the output matrix.
    return output;
}

template <typename T>
Matrix<T>& Matrix<T>::operator+= (const Matrix<T> & matrix2) {
    /**
     * function operator+=
     * Uses Matrix<T>::operator+ to add to a matrix.
     *
     * Overrides the += operator for a Matrix object.
     *
     * Parameters:
     *   Matrix<T> matrix2: A second matrix to add to the first, of the same type T.
     *
     * Returns:
     *   Matrix<T>&: A reference to an updated version of the matrix.
     */

    // Use the overridden matrix addition operator to add the matrices.
    // The 'this' keyword is a pointer that must be unpacked before addition.
    *this = *this + matrix2;

    // Return the matrix.
    return *this;
}

template <typename T>
Matrix<T> Matrix<T>::operator- (const Matrix<T> & matrix2){
    /**
     * function operator-
     * Subtracts two matrices.
     *
     * Overrides the - operator for two Matrix objects of the same type T.
     *
     * Parameters:
     *   const Matrix<T> matrix2: A second matrix to subtract from the first, of the same type T.
     *
     * Returns:
     *   Matrix<T>: The difference between the two matrices.
     *
     * Errors:
     *   Throws an error if the matrices are different shapes.
     */

    // Data validation.
    try {
        // Check that the matrices are the same shape.
        if (!(rows == matrix2.rows && cols == matrix2.cols)) {
            // Throw an error.
            throw 1;
        }
    }
    catch (int e) {
        // Print an error statement.
        std::cout << "Matrix subtraction: matrices different shapes." << std::endl;
        // Exit the program and report the failure.
        exit(EXIT_FAILURE);
    }

    // Create output matrix.
    Matrix<T> output(cols, rows);

    // Loop through the rows of the matrix.
    for (unsigned i = 0; i < rows; ++i){
        // Loop through the items in each row.
        for (unsigned j = 0; j < cols; ++j){
            // Find the difference between the values of the matrices at the current index.
            output[i][j] = matrix[i][j] - matrix2[i][j];
        }
    }

    // Return the output matrix.
    return output;
}

template <typename T>
Matrix<T>& Matrix<T>::operator-= (const Matrix<T> & matrix2) {
    /**
     * function operator-=
     * Uses Matrix<T>::operator- to add to a matrix.
     *
     * Overrides the -= operator for a Matrix object.
     *
     * Parameters:
     *   Matrix<T> matrix2: A second matrix to subtract from the first, of the same type T.
     *
     * Returns:
     *   Matrix<T>&: A reference to an updated version of the matrix.
     */

     // Use the overridden matrix addition operator to subtract the matrices.
     // The 'this' keyword is a pointer that must be unpacked before subtraction.
    *this = *this - matrix2;

    // Return the matrix.
    return *this;
}

template <typename T>
Matrix<T> Matrix<T>::operator* (const Matrix<T> & matrix2) {
    /**
     * function operator *
     * Multiplies two matrices.
     *
     * Overrides the * operator for two Matrix objects of the same type T.
     *
     * Parameters:
     *   Matrix<T> matrix2: A second matrix of the same type T to multiply with the first.
     *
     * Returns:
     *   Matrix<T>: The product of the two matrices.
     *
     * Errors:
     *   Throws an error if the sizes are such that multiplication cannot occur.
    */

    // Data validation.
    try {
        // Check that the number of rows in matrix2 is the same as the number of
        // columns in the original matrix.
        if (cols != matrix2.rows) {
            throw 1;
        }
    }
    catch (int e) {
        // Print an error statenent.
        std::cout << "Matrix multiplication: matrices inconsistent sizes." << std::endl;
        // Exit the program and report the failure.
        exit(EXIT_FAILURE);
    }

    // Create output matrix.
    Matrix<T> output(matrix2.cols, rows);

    // Loop through the rows of the output matrix.
    for (unsigned i = 0; i < output.rows; ++i){
      // Loop through each item in the row.
        for (unsigned j = 0; j < output.cols; ++j){
            // Find the value of the output matrix using the definition of
            // matrix multiplication.
            T sum = 0;
            for (unsigned k = 0; k < cols; ++k){
                sum += matrix[i][k] * matrix2[k][j];
            }
            output[i][j] = sum;
        }
    }

    // Return the output matrix.
    return output;
}

template <typename T>
Matrix<T>& Matrix<T>::operator*= (const Matrix<T> & matrix2) {
    /**
     * function operator*=
     * Uses Matrix<T>::operator* to multiply and update a matrix.
     *
     * Overrides the *= operator for a matrix object.
     *
     * Parameters:
     *   Matrix<T> matrix2: A second matrix to multiply with the first, of the same type T.
     *
     * Returns:
     *   Matrix<T>&: A reference to an updated version of the matrix.
    */

    // Use the overridden matrix addition operator to add the matrices.
    // The 'this' keyword is a pointer that must be unpacked before addition.
    *this = *this * matrix2;

    // Return the matrix.
    return *this;
}

template <typename T>
Matrix<T> Matrix<T>::operator* (T value) {
    /**
     * function operator *
     * Multiplies a matrix by some scalar of type T.
     *
     * Overrides the * operator for a Matrix objects of type T and a literal of the same type T.
     *
     * Parameters:
     *   T value: A value of type T to multiply with the matrix.
     *
     * Returns:
     *   Matrix<T>: The scalar product of the matrix and the scalar.
    */

    // Create output matrix.
    Matrix<T> output(cols, rows);

    // Loop through the rows of the output matrix.
    for (unsigned i = 0; i < output.rows; ++i) {
        // Loop through each item in the row.
        for (unsigned j = 0; j < output.cols; ++j) {
            // Multiply the matrix at the current index by the input value.
            output[i][j] = matrix[i][j] * value;
        }
    }

    // Return the output matrix.
    return output;
}

template <typename T>
Matrix<T>& Matrix<T>::operator*= (T value) {
    /**
     * function operator*=
     * Uses Matrix<T>::operator* to multiply and update a matrix.
     *
     * Overrides the *= operator for a matrix object for scalar multiplication.
     *
     * Parameters:
     *   T value: A value of type T to multiply with the matrix.
     *
     * Returns:
     *   Matrix<T>&: A reference to an updated version of the matrix.
    */

    // Use the overridden matrix addition operator to add the matrices.
    // The 'this' keyword is a pointer that must be unpacked before addition.
    *this = *this * value;

    // Return the matrix.
    return *this;
}

template <typename T>
Matrix<T> Matrix<T>::operator/ (T value) {
    /**
     * function operator /
     * Divides a matrix by some scalar of type T.
     *
     * Overrides the / operator for a Matrix objects of type T and a literal of the same type T.
     *
     * Parameters:
     *   T value: A value of type T by which to divide the matrix.
     *
     * Returns:
     *   Matrix<T>: The scalar product of the matrix and the reciprocal of the scalar.
    */

    // Create output matrix.
    Matrix<T> output(cols, rows);

    // Loop through the rows of the output matrix.
    for (unsigned i = 0; i < output.rows; ++i) {
        // Loop through each item in the row.
        for (unsigned j = 0; j < output.cols; ++j) {
            // Multiply the matrix at the current index by the input value.
            output[i][j] = matrix[i][j] / value;
        }
    }

    // Return the output matrix.
    return output;
}

template <typename T>
Matrix<T>& Matrix<T>::operator/= (T value) {
    /**
     * function operator*=
     * Uses Matrix<T>::operator/ to divide and update a matrix.
     *
     * Overrides the /= operator for a matrix object for scalar division.
     *
     * Parameters:
     *   T value: A value of type T to divide the matrix by.
     *
     * Returns:
     *   Matrix<T>&: A reference to an updated version of the matrix.
    */

    // Use the overridden matrix addition operator to add the matrices.
    // The 'this' keyword is a pointer that must be unpacked before addition.
    *this = *this / value;

    // Return the matrix.
    return *this;
}

#endif //MATRIX_H

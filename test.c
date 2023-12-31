#include "greatest/greatest.h"
#include "vector.h"

VECTOR_INIT(test_vector, int)


TEST test_vector_resizing(void) {
    test_vector *v = test_vector_new();
    ASSERT_EQ(v->m, DEFAULT_VECTOR_SIZE);
    ASSERT_EQ(v->n, 0);

    for (int i = 0; i < 10; i++) {
        test_vector_push(v, i);
    }
    size_t expected_size = DEFAULT_VECTOR_SIZE * 3 / 2;
    ASSERT_EQ(v->m, expected_size);
    ASSERT_EQ(v->n, 10);

    for (int i = 0; i < 10; i++) {
        ASSERT_EQ(v->a[i], i);
    }

    test_vector *w = test_vector_new_size(16);
    ASSERT_EQ(w->m, 16);
    ASSERT_EQ(w->n, 0);

    for (int i = 0; i < 17; i++) {
        test_vector_push(w, i);
    }
    ASSERT_EQ(w->m, 16 * 3 / 2);
    ASSERT_EQ(w->n, 17);

    test_vector_extend(v, w);
    expected_size = expected_size * 3 / 2 * 3 /2;
    ASSERT_EQ(v->m, expected_size);
    ASSERT_EQ(v->n, 27);

    test_vector_destroy(v);
    PASS();
}

/* Add definitions that need to be in the test runner's main file. */
GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
    GREATEST_MAIN_BEGIN();      /* command-line options, initialization. */

    RUN_TEST(test_vector_resizing);

    GREATEST_MAIN_END();        /* display results */
}

from sklearn import decomposition


def boston_pca(x, n_components=1):
    print('PCA')
    n_components = n_components
    print('Components: %i\n' % n_components)
    pca = decomposition.PCA(n_components)
    x = x
    pca.fit(x)
    print('PCA components: %s' % pca.components_)
    print('PCA variance ratio: %s' % pca.explained_variance_ratio_)
    x_decomposed = pca.transform(x)

    for i in range(0, 10):
        print('\n%i.' % (i+1))
        print('Decomposed X: %s' % x_decomposed[i])

    return x_decomposed

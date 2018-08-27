"""
Hierarchical Text Classifier interface.
"""
from networkx import DiGraph, is_tree
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_array, check_consistent_length, check_is_fitted, check_X_y
from sklearn.utils.multiclass import check_classification_targets

from hierarchical_text_classification.htc.constant import *
from hierarchical_text_classification.htc.dummy_progress import DummyProgress
from hierarchical_text_classification.htc.util import *


class HierarchicalTextClassifier(BaseEstimator, ClassifierMixin, MetaEstimatorMixin):
    """
    Implementation of Hierarchical Text Classifier [1].

    The hierarchical classifier framework is developed based on github project sklearn-hierarchical-classification [2].

    References:
    -----------
    [1] Sun, A., & Lim, E. (2001). Hierarchical Text Classification and Evaluation.
    Proceedings of the 2001 IEEE International Conference on Data Mining, (November), 521–528.
    https://doi.org/10.1109/ICDM.2001.989560
    [2] Globality-corp. SKlearn Hierarchical Classification. Available from
    https://github.com/globality-corp/sklearn-hierarchical-classification, Aug. 2018.

    Parameters:
    ------------
    base_estimator : classifier object, function, dict, or None
        A scikit-learn compatible classifier object implementing 'fit' and 'predict_proba' to be used as the
        base classifier.
        If a callable function is given, it will be called to evaluate which classifier to instantiate for
        current node. The function will be called with the current node_id and the graph instance.
        Alternatively, a dictionary mapping classes to classifier objects can be given. The value of the
        dict is a classifier object. The key of the dict could be a tuple of (node_id, clf_type) or just
        node_id for all the types. clf_type is a string, where clf_local' indicates a local classifier and
         'clf_subtree' represents a subtree classifier. A special key 'default' can be set for matching
         other keys that are not in the dict.

    class_hierarchy : networkx.DiGraph object, or dict-of-dicts adjacency representation (see examples)
        A directed graph which represents the target classes and their relations. Must be a tree/DAG (no cycles).
        If not provided, this will be initialized during the `fit` operation into a trivial graph structure linking
        all classes given in `y` to an artificial "ROOT" node.

    root : integer or string, default -1
        The unique identifier for the qualified root node in the class hierarchy. The hierarchical classifier
        assumes that the given class hierarchy graph is a rooted tree, it has a single designated root node
        of in-degree 0.

    labels : Array of labels, integer or string
        The labels to be classified, the sequence of labels must be consistent with the training and predicting label y,
        which is in the shape of [:, `n_classes`]. If it is None, labels array will be generated via iterating the class
        tree class_hierarchy.

    subtree_threshold : Float
        Threshold of the probability for local classifiers

    local_threshold : Float
        Threshold of the probability for subtree classifiers

    progress_wrapper : progress generator or None
        If value is set, will attempt to use the given generator to display progress updates. This added functionality
        is especially useful within interactive environments (e.g in a testing harness or a Jupyter notebook). Setting
        this value will also enable verbose logging. Common values in tqdm are `tqdm_notebook` or `tqdm`

    Attributes
    ----------
    classes_ : array, shape = [`n_classes`]
        Flat array of class labels
    """

    def __init__(self, base_estimator=None, class_hierarchy=None, root=ROOT, labels=None,
                 subtree_threshold=0.5, local_threshold=0.5, progress_wrapper=None):
        self.base_estimator = base_estimator
        self.class_hierarchy = class_hierarchy
        self.root = root
        self.classes_ = labels
        self.subtree_threshold = subtree_threshold
        self.local_threshold = local_threshold
        self.progress_wrapper = progress_wrapper

    def fit(self, X, y=None, sample_weight=None):
        """Fit underlying classifiers.

        Parameters
        ----------
        X : (sparse) array-like, shape = [n_samples, n_features]
            Data.

        y : (sparse) array-like, shape = [n_samples, ], [n_samples, n_classes]
            Multi-class targets. An indicator matrix turns on multi-label classification.

        sample_weight : array-like, shape (n_samples,), optional (default=None)
            Weights applied to individual samples (1. for unweighted).

        Returns
        -------
        self

        """
        X, y = check_X_y(X, y, accept_sparse='csr', multi_output=True)
        check_classification_targets(y)
        if sample_weight is not None:
            check_consistent_length(y, sample_weight)

        # Check that parameter assignment is consistent
        self._check_parameters()

        # Initialize NetworkX Graph from input class hierarchy
        self.class_hierarchy_ = self.class_hierarchy or make_flat_hierarchy(list(np.unique(y)), root=self.root)
        self.graph_ = DiGraph(self.class_hierarchy_)

        if not is_tree(self.graph_):
            raise ValueError("Given class hierarchy is not tree structure")

        if self.classes_ is None:
            self.classes_ = list(
                node
                for node in self.graph_.nodes()
                if node != self.root
            )

        # Convert y to sparse matrix, the label sequence shall be the same with self.classes_
        if len(y.shape) == 1:
            y = convert_label_vector_to_csr_matrix(y, self.classes_)
        elif len(y.shape) == 2:
            y = csr_matrix(y)
            if self.n_classes_ != y.shape[1]:
                raise ValueError("Shape of y: %s is inconsistent with # of classes %s" % (y.shape, self.n_classes_))
        else:
            raise ValueError("Unsupported shape of y: %s" % y.shape)

        # Recursively build coverage sets for the local classifier of each node in graph
        with self._progress(total=self.n_classes_ + 1, desc="Building coverage sets") as progress:
            logging.info("recursively building coverage sets for the local classifier of each node in graph")
            self._recursive_build_coverage_sets(X, y, node_id=self.root, progress=progress)

        # Build training feature sets for the local classifier of each node in graph
        with self._progress(total=self.n_classes_ + 1, desc="Building local features") as progress:
            logging.info("building training feature sets for the local classifier of each node in graph")
            self._build_local_train_data(X, y, progress=progress)

        # Build training feature sets for the subtree classifier of each node in graph
        with self._progress(total=self.n_classes_ + 1, desc="Building subtree features") as progress:
            logging.info("building training feature sets for the subtree classifier of each node in graph")
            self._build_subtree_train_data(X, y, progress=progress)

        # Recursively train local classifiers
        with self._progress(total=self.n_classes_ + 1, desc="Training local classifiers") as progress:
            logging.info("recursively training local classifiers")
            self._train_local_classifiers(X, y, progress=progress)

        # Recursively train subtree classifiers
        with self._progress(total=self.n_classes_ + 1, desc="Training subtree classifiers") as progress:
            logging.info("recursively training subtree classifiers")
            self._train_subtree_classifiers(X, y, progress=progress)

        return self

    def predict(self, X, output="matrix"):
        """Predict multi-class targets using underlying estimators.

        Parameters
        ----------
        X : (sparse) array-like, shape = [n_samples, n_features]
            Data.

        output : string, the output structure which can be `vector` or `matrix`
                 Only single-label prediction can be output as vector type.

        Returns
        -------
        y : (sparse) array-like, shape = [n_samples, ], [n_samples, n_classes].
            Predicted multi-class targets.

        """
        check_is_fitted(self, "graph_")
        X = check_array(X, accept_sparse="csr")

        def _classify(x):
            preds = np.zeros(self.n_classes_)
            scores = np.zeros(self.n_classes_)
            self._recursive_predict(x, self.root, preds, scores)
            return preds

        y_pred = apply_along_rows(_classify, X=X)

        if output == "vector":
            y_pred = convert_label_matrix_to_vector(y_pred, self.classes_)

        return y_pred

    def predict_proba(self, X):
        """
        Return probability estimates for the test vector X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array-like, shape = [n_samples, n_classes]
            Returns the probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute `classes_`.
        """
        check_is_fitted(self, "graph_")
        X = check_array(X, accept_sparse="csr")

        def _classify(x):
            preds = np.zeros(self.n_classes_)
            scores = np.zeros(self.n_classes_)
            self._recursive_predict(x, self.root, preds, scores)
            return scores

        y_prob = apply_along_rows(_classify, X=X)
        return y_prob

    def _build_local_train_data(self, X, y, progress):
        # TODO add negative sampling for balancing data
        for node_id in self.graph_.nodes:
            logging.debug("Building local features for node: %s", node_id)
            progress.update(1)
            # Root node
            if self.graph_.in_degree(node_id) == 0:
                # Because all nodes are constructed with a virtual root node,
                # thus we don't need to train a local classifier for the virtual root node
                continue
            # Leaf node
            elif self.graph_.out_degree(node_id) == 0:
                pos_features = self._extract_node_data(X, y, node_id)
                parent_id = self._get_parent_node_id(node_id)
                neg_features = self.graph_.node[parent_id][COVERAGE] - pos_features
            # Internal node
            else:
                pos_features = self._extract_node_data(X, y, node_id)
                neg_features = self.graph_.node[node_id][COVERAGE] - pos_features

            # Build features and targets
            features, targets = build_binary_training_data(pos_features, neg_features)
            self.graph_.node[node_id][LOCAL_DATA] = (features, targets)

            # Build and store meta-data for node
            self.graph_.node[node_id][LOCAL_META] = self._build_metafeatures(
                len(targets[targets == 1]), len(targets[targets == -1]))

    def _build_subtree_train_data(self, X, y, progress):
        # TODO add negative sampling for balancing data
        for node_id in self.graph_.nodes:
            logging.debug("Building subtree features for node: %s", node_id)
            progress.update(1)
            # Root node
            if self.graph_.in_degree(node_id) == 0:
                # Because all nodes are constructed with a virtual root node,
                # thus we don't need to train a subtree classifier for the virtual root node
                continue
            # Leaf node
            elif self.graph_.out_degree(node_id) == 0:
                # No subtree classifier for leaf nodes
                continue
            # Internal node
            else:
                pos_features = self.graph_.node[node_id][COVERAGE]
                parent_id = self._get_parent_node_id(node_id)
                parent_data = self._extract_node_data(X, y, parent_id)
                neg_features = self.graph_.node[parent_id][COVERAGE] - pos_features - parent_data

            # Build features and targets
            features, targets = build_binary_training_data(pos_features, neg_features)
            self.graph_.node[node_id][SUBTREE_DATA] = (features, targets)

            # Build and store meta-data for node
            self.graph_.node[node_id][SUBTREE_META] = self._build_metafeatures(
                len(targets[targets == 1]), len(targets[targets == -1]))

    def _train_local_classifiers(self, X, y, progress):
        for node_id in self.graph_.nodes:
            progress.update(1)
            # Root node
            if self.graph_.in_degree(node_id) == 0:
                # Because all nodes are constructed with a virtual root node,
                # thus we don't need to train a local classifier for the virtual root node
                continue
            # Train for internal and leaf nodes
            logging.debug("Training local classifier for node: %s", node_id)
            features, targets = self.graph_.node[node_id][LOCAL_DATA]
            self.graph_.node[node_id][LOCAL_CLF] = self._train_classifier(features, targets, node_id, CLF_LOCAL_TYPE)

    def _train_subtree_classifiers(self, X, y, progress):
        for node_id in self.graph_.nodes:
            progress.update(1)
            # Root node
            if self.graph_.in_degree(node_id) == 0:
                # Because all nodes are constructed with a virtual root node,
                # thus we don't need to train a local classifier for the virtual root node
                continue
            elif self.graph_.out_degree(node_id) == 0:
                # No subtree classifier for leaf nodes
                continue
            # Train for internal nodes
            logging.debug("Training local classifier for node: %s", node_id)
            features, targets = self.graph_.node[node_id][SUBTREE_DATA]
            self.graph_.node[node_id][SUBTREE_CLF] = self._train_classifier(features, targets, node_id, CLF_SUBTREE_TYPE)

    def _build_metafeatures(self, n_positive, n_negative):
        return dict(
            n_positive=n_positive,
            n_negative=n_negative)

    def _recursive_predict(self, x, node_id, preds, scores):

        def inner_local_predict(inner_local_clf, x):
            probs = inner_local_clf.predict_proba(x)[0]
            if isinstance(inner_local_clf, DummyClassifier):
                inner_local_pred = inner_local_clf.predict(x)[0]
                if inner_local_pred == 1:
                    inner_prob_one = probs
                else:
                    inner_prob_one = 1 - probs
            else:
                inner_prob_one = inner_local_clf.predict_proba(x)[0][1]
                if inner_prob_one > self.local_threshold:
                    inner_local_pred = 1
                else:
                    inner_local_pred = -1
            return inner_local_pred, inner_prob_one

        # Root node
        if self.graph_.in_degree(node_id) == 0:
            for child in self.graph_.successors(node_id):
                self._recursive_predict(x, child, preds, scores)
            return
        label_idx = self.find_label_index_(node_id)
        # Leaf node
        if self.graph_.out_degree(node_id) == 0:
            local_clf = self.graph_.node[node_id][LOCAL_CLF]
            local_pred, prob_one = inner_local_predict(local_clf, x)
            scores[label_idx] = prob_one
            if local_pred == 1:
                preds[label_idx] = local_pred
            return
        # Internal node
        else:
            # Predict whether in this subtree
            subtree_clf = self.graph_.node[node_id][SUBTREE_CLF]
            if subtree_clf is None:
                logging.debug("no subtree classifier for node: %s", node_id)
                return
            if isinstance(subtree_clf, DummyClassifier):
                subtree_pred = subtree_clf.predict(x)[0]
            else:
                subtree_prob_one = subtree_clf.predict_proba(x)[0][1]
                if subtree_prob_one > self.subtree_threshold:
                    subtree_pred = 1
                else:
                    subtree_pred = -1
            # In this subtree, predict whether in this local node
            if subtree_pred == 1:
                local_clf = self.graph_.node[node_id][LOCAL_CLF]
                if local_clf is None:
                    logging.debug("no local classifier for node: %s", node_id)
                    return
                local_pred, prob_one = inner_local_predict(local_clf, x)
                scores[label_idx] = prob_one
                # In this local node, add this label and return
                if local_pred == 1:
                    preds[label_idx] = local_pred
                    return
                # Not in this local node but still in this subtree, recursively find labels in the successors
                else:
                    for child in self.graph_.successors(node_id):
                        self._recursive_predict(x, child, preds, scores)
                    return
            else:
                return

    def _train_classifier(self, X, y, node_id, clf_type):
        num_targets = len(np.unique(y))

        logging.debug(
            "_train_classifier() - Training local classifier for node: %s, X_.shape: %s, len(y): %s, n_targets: %s",
            node_id,
            X.shape,
            len(y),
            num_targets,
        )

        if X.shape[0] == 0:
            # No training data could be materialized for current node
            logging.warning(
                "_train_classifier() - no training data available at node %s",
                node_id,
            )
            return
        elif num_targets == 1:
            # Training data could be materialized for only a single target at current node
            constant = y[0]
            logging.debug(
                "_train_classifier() - only one single target (child node) available to train classifier for node %s",
                node_id,
                constant,
            )

            clf = DummyClassifier(strategy="constant", constant=constant)
        else:
            clf = self._base_estimator_for(node_id, clf_type)

        clf.fit(X=X, y=y)
        return clf

    def _should_early_terminate(self, current_node, prediction, score):
        pass

    def _recursive_build_coverage_sets(self, X, y, node_id, progress):
        logging.debug("building coverage sets for node " + node_id)

        if COVERAGE in self.graph_.node[node_id]:
            # Already visited this node in feature building phase
            return self.graph_.node[node_id][COVERAGE]

        logging.debug("building coverage features for node: %s", node_id)
        progress.update(1)

        # Add self except root node
        self.graph_.node[node_id][COVERAGE] = self._extract_node_data(X, y, node_id)

        # Leaf node
        if self.graph_.out_degree(node_id) == 0:
            return self.graph_.node[node_id][COVERAGE]

        # Non-leaf node, add successors
        for child_node_id in self.graph_.successors(node_id):
            self.graph_.node[node_id][COVERAGE] += \
                self._recursive_build_coverage_sets(
                    X=X,
                    y=y,
                    node_id=child_node_id,
                    progress=progress,
                )

        return self.graph_.node[node_id][COVERAGE]

    def _extract_node_data(self, X, y, node_id):
        if node_id == self.root:
            return csr_matrix(X.shape)
        idx = self.find_label_index_(node_id)
        nnz_row, nnz_col = y.nonzero()
        indices = nnz_row[nnz_col == idx]
        return extract_rows_csr(X, indices)

    @property
    def n_classes_(self):
        return len(self.classes_)

    def find_label_index_(self, label):
        idx = find_array_index(self.classes_, label)
        if idx == -1:
            raise ValueError("Cannot find label index for label: " + str(label))
        return idx

    def _check_parameters(self):
        """Check the parameter assignment is valid and internally consistent."""
        # TODO
        return True

    def _base_estimator_for(self, node_id, clf_type):
        base_estimator = None
        if not self.base_estimator:
            # No base estimator specified by user, try to pick best one
            base_estimator = self._make_base_estimator(node_id)

        elif isinstance(self.base_estimator, dict):
            # User provided dictionary mapping nodes to estimators
            # TODO differentiate local and subtree classifiers
            if (node_id, clf_type) in self.base_estimator:
                base_estimator = self.base_estimator[(node_id, clf_type)]
            elif node_id in self.base_estimator:
                base_estimator = self.base_estimator[node_id]
            else:
                base_estimator = self.base_estimator[DEFAULT]

        elif is_estimator(self.base_estimator):
            # Single base estimator object, return a copy
            base_estimator = self.base_estimator

        else:
            # By default, treat as callable factory
            base_estimator = self.base_estimator(node_id=node_id, graph=self.graph_)

        return clone(base_estimator)

    def _make_base_estimator(self, node_id):
        logging.info("using the default base estimator -- logistic regression ..")
        return LogisticRegression()

    def _get_parent_node_id(self, node_id):
        for i in self.graph_.predecessors(node_id):
            return i

    def _progress(self, total, desc, **kwargs):
        if self.progress_wrapper:
            return self.progress_wrapper(total=total, desc=desc)
        else:
            return DummyProgress()

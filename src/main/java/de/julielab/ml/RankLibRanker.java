package de.julielab.ml;

import cc.mallet.types.*;
import ciir.umass.edu.features.*;
import ciir.umass.edu.learning.*;
import ciir.umass.edu.metric.METRIC;
import ciir.umass.edu.metric.MetricScorer;
import ciir.umass.edu.metric.MetricScorerFactory;
import de.julielab.java.utilities.FileUtilities;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

public class RankLibRanker {
    private final static Logger log = LoggerFactory.getLogger(RankLibRanker.class);
    private final MetricScorerFactory metricScorerFactory;
    private Ranker ranker;
    private RANKER_TYPE rType;
    private int[] features;
    private METRIC trainMetric;
    private int k;
    private Normalizer featureNormalizer;

    /**
     * <p>Creates an object that has all information to create a RankLib ranker but does not immediately do it.</p>
     * <p>The actual ranker is create by using the {@link #train(InstanceList)}</p>
     *
     * @param rType       The type of ranker to create.
     * @param features    The feature indices the ranker should be trained with and which should be used for ranking.
     * @param trainMetric The metric to be optimized for during training.
     * @param k           The top-number of documents to be used for the training metric.
     * @param normalizer  The feature normalizer to use, may be null. Possible values are <tt>sum</tt>, <tt>zscore</tt> and <tt>linear</tt>.
     */
    public RankLibRanker(RANKER_TYPE rType, int[] features, METRIC trainMetric, int k, String normalizer) {
        this.rType = rType;
        this.features = features;
        this.trainMetric = trainMetric;
        this.k = k;
        metricScorerFactory = new MetricScorerFactory();
        // This causes the RankLib datapoints to return 0f for feature values they don't have. We need this because
        // in the original RankLib, there was a static field enumerating all known features loaded within the current JVM.
        // I removed that because it is not thread safe.
        DataPoint.missingZero = true;
        if (normalizer != null) {
            if (normalizer.equalsIgnoreCase("sum"))
                featureNormalizer = new SumNormalizor();
            else if (normalizer.equalsIgnoreCase("zscore"))
                featureNormalizer = new ZScoreNormalizor();
            else if (normalizer.equalsIgnoreCase("linear"))
                featureNormalizer = new LinearNormalizer();
            else {
                throw new IllegalArgumentException("Unknown normalizer: " + normalizer);
            }
        }
    }

    public double score(InstanceList documentList, METRIC scoringMetric, int k) {
        final MetricScorer scorer = metricScorerFactory.createScorer(scoringMetric, k);
        final Map<String, RankList> rankLists = convertToRankList(documentList);
        return scorer.score(rankLists.values().stream().collect(Collectors.toList()));
    }

    public void train(InstanceList documents) {
        log.info("Training on {} documents without validation set.", documents.size());
        final Map<String, RankList> rankLists = convertToRankList(documents);
        this.features = this.features != null ? this.features : FeatureManager.getFeatureFromSampleVector(new ArrayList(rankLists.values()));
        ranker = new RankerTrainer().train(rType, new ArrayList(rankLists.values()), features, metricScorerFactory.createScorer(trainMetric, k));
    }

    /**
     * <p>Trains a ranker on basis of the delivered MALLET instances.</p>
     * <p>MALLET instance have multiple fields that can be set freely. Those fields translate to the requirements
     * of the RankLib library as follows:
     * <ul>
     *     <li>{@link Instance#getName()} - query ID</li>
     *     <li>{@link Instance#getSource()} - {@link DataPoint#getDescription()}, used for the document ID</li>
     *     <li>{@link Instance#getData()} - must be a MALLET {@link FeatureVector} that will be translated into RankLib {@link DataPoint} objects.</li>
     * </ul>
     * In this method, the instances will be grouped by their name a.k.a their queryId. Thus, the queryId must represent
     * the different lists of instances to be ranked against each other.
     * </p>
     *
     * @param documents    The MALLET instances with their features set.
     * @param doValidation Whether to use a validation set to select the ultimate model.
     * @param fraction     The fraction of data used for the training set.
     * @param randomSeed   The seed used to shuffle the data before splitting into train-dev sets.
     */
    public void train(InstanceList documents, boolean doValidation, float fraction, int randomSeed) {
        if (!doValidation)
            log.info("Training on {} documents without validation set.", documents.size());
        else
            log.info("Training on {} documents where a fraction of {} is used for training and the rest for validation. The split is done randomly with a seed of {}.", documents.size(), fraction, randomSeed);
        final Map<String, RankList> rankLists = convertToRankList(documents);
        if (featureNormalizer != null)
            rankLists.values().forEach(featureNormalizer::normalize);
        List<RankList> train;
        List<RankList> validation;
        if (doValidation) {
            final Pair<Map<String, RankList>, Map<String, RankList>> trainValData = makeValidationSplit(rankLists, fraction, randomSeed);
            train = new ArrayList(trainValData.getLeft().values());
            validation = new ArrayList<>(trainValData.getRight().values());
        } else {
            train = new ArrayList<>(rankLists.values());
            validation = Collections.emptyList();
        }

        this.features = this.features != null ? this.features : FeatureManager.getFeatureFromSampleVector(new ArrayList(rankLists.values()));
        ranker = new RankerTrainer().train(rType, train, validation, features, metricScorerFactory.createScorer(trainMetric, k));
        if (!documents.isEmpty()) {
            log.trace("LtR features: " + documents.getAlphabet());
        }
    }

    private Pair<Map<String, RankList>, Map<String, RankList>> makeValidationSplit(Map<String, RankList> allData, float fraction, int randomSeed) {
        if (fraction < 0 || fraction >= 1)
            throw new IllegalArgumentException("The fraction to be taken from the training data for validation is specified as " + fraction + " but it must be in [0, 1).");
        int size = (int) (fraction * allData.size());
        log.info("Splitting into training size of {} and validation size of {} queries", size, allData.size() - size);
        final List<RankList> shuffledData = new ArrayList<>(allData.values());
        Collections.shuffle(shuffledData, new Random(randomSeed));
        Map<String, RankList> train = new HashMap<>();
        Map<String, RankList> val = new HashMap<>();
        for (int i = 0; i < size; i++)
            train.put(shuffledData.get(i).getID(), shuffledData.get(i));
        for (int i = size; i < shuffledData.size(); i++)
            val.put(shuffledData.get(i).getID(), shuffledData.get(i));
        return new ImmutablePair<>(train, val);
    }

    /**
     * <p>Creates RankLib {@link SparseDataPoint} objects for each documents and groups them by query ID.</p>
     *
     * @param documents The documents to convert into the RankLib format.
     * @return A map where each query ID occurring in the input documents is mapped to the RankList made from the documents for this query ID.
     */
    private Map<String, RankList> convertToRankList(InstanceList documents) {
        final LinkedHashMap<String, List<DataPoint>> dataPointsByQueryId = documents.stream().map(d -> {
            final FeatureVector fv = (FeatureVector) d.getData();
            if (fv == null)
                throw new IllegalArgumentException("Cannot train a ranker because the input documents have no feature vector.");
            final double[] values = fv.getValues();
            final int[] indices = fv.getIndices();

            float[] ranklibValues = new float[fv.numLocations()];
            int[] ranklibIndices = new int[fv.numLocations()];
            if (values == null) {
                // binary vector
                Arrays.fill(ranklibValues, 1f);
            } else {
                for (int i = 0; i < fv.numLocations(); i++)
                    ranklibValues[i] = (float) values[i];
            }
            for (int i = 0; i < fv.numLocations(); i++)
                // RankLib indices start counting at 1, MALLET at 0
                ranklibIndices[i] = indices[i] + 1;
            String queryId = d.getName().toString();
            DataPoint dp = new SparseDataPoint(ranklibValues, ranklibIndices, queryId, (Float)((Label) d.getTarget()).getEntry());
            // The description field of the DataPoint is used to store the document ID
            dp.setDescription("#"+d.getSource().toString());
            return dp;
        }).collect(Collectors.groupingBy(DataPoint::getID, LinkedHashMap::new, Collectors.toList()));
        final LinkedHashMap<String, RankList> rankLists = new LinkedHashMap<>();
        dataPointsByQueryId.forEach((key, value) -> rankLists.put(key, new RankList(value)));
        return rankLists;
    }

    public void load(File modelFile) throws IOException {
        try (final BufferedReader br = FileUtilities.getReaderFromFile(modelFile)) {
            final String model = br.lines().collect(Collectors.joining(System.getProperty("line.separator")));
            ranker = new RankerFactory().loadRankerFromString(model);
        }
    }

    public void save(File modelFile) {
        if (!modelFile.getParentFile().exists())
            modelFile.getParentFile().mkdirs();
        ranker.save(modelFile.getAbsolutePath());
    }

    /**
     * RankLib models are stored as strings listing the model parameters. This method can be used to return this exact string.
     *
     * @return The model data.
     */
    public String getModelAsString() {
        return ranker.model();
    }

    /**
     * RankLib models are stored as string listing the model parameters. Such a string can be retrieved from {@link #getModelAsString()}.
     * Passing that string to this method sets the ranker to the given parameters.
     *
     * @param modelString The model data.
     */
    public void loadFromString(String modelString) {
        ranker = new RankerFactory().loadRankerFromString(modelString);
    }

    /**
     * <p>Ranks a list of MALLET {@link Instance} objects according to a previously trained or loaded model.</p>
     * <p>MALLET instance have multiple fields that can be set freely. Those fields translate to the requirements
     * of the RankLib library as follows:
     * <ul>
     *     <li>{@link Instance#getName()} - query ID</li>
     *     <li>{@link Instance#getSource()} - {@link DataPoint#getDescription()}, used for the document id</li>
     *     <li>{@link Instance#getData()} - must be a MALLET {@link FeatureVector} that will be translated into RankLib {@link DataPoint} objects.</li>
     * </ul>
     * The output is a new <tt>InstanceList</tt> sorted by descreasing ranking scores. The score itsel is stored in the
     * 'score' property of the {@link Instance#getProperty(String)} map.
     * </p>
     */
    public InstanceList rank(InstanceList documents) {

        // here, the keys must be the same as in the final Instance doc = docsById.get(dp.getID() + docId) line below
        // and thus must be the same as in convertToRankList() where the DataPoints are created
        Map<String, Instance> docsById = documents.stream().collect(Collectors.toMap(d -> d.getName().toString() + "#"+d.getSource().toString(), Function.identity()));

        if (docsById.size() != documents.size())
            throw new IllegalArgumentException("The passed documents do not have unique IDs. The input document list has size " + documents + ", its ID map form only " + docsById.size());

        final Map<String, RankList> rankLists = convertToRankList(documents);
        if (featureNormalizer != null)
            rankLists.values().forEach(featureNormalizer::normalize);
        for (RankList rl : rankLists.values()) {
            for (int i = 0; i < rl.size(); i++) {
                final DataPoint dp = rl.get(i);
                final double score = ranker.eval(dp);
                final String docId = dp.getDescription();
                final Instance doc = docsById.get(dp.getID() + docId);
                doc.setProperty("score", score);
            }
        }

        final InstanceList ret = new InstanceList(documents.getDataAlphabet(), documents.getTargetAlphabet());
        ret.addAll(documents);
        Collections.sort(ret, Comparator.<Instance>comparingDouble(d -> (double) d.getProperty("score")).reversed());
        return ret;
    }


    public Ranker getRankLibRanker() {
        return ranker;
    }

    public static InstanceList loadSvmLightData(File dataFile) throws Exception {
        // For MALLET, we create one feature or data alphabet and the target or label alphabet
        // to encode the feature names into feature indexes and the relevance into label objects.
        Alphabet dataAlphabet = new Alphabet();
        LabelAlphabet targetAlphabet = new LabelAlphabet();
        // This instance list will only accept instances with the correct alphabets.
        InstanceList ret = new InstanceList(dataAlphabet, targetAlphabet);
        try (BufferedReader br = FileUtilities.getReaderFromFile(dataFile)) {
            int documentId = 0;
            for (String line : (Iterable<String>) () -> br.lines().iterator()) {
                String[] split = line.split("\\s+");
                Float relevance = Float.parseFloat(split[0]);
                String queryId = split[1];
                int[] indices = new int[5];
                double[] features = new double[5];
                // The first position is the relevance score, the second is the query ID.
                // After the hash character there comes the optional document ID.
                boolean hasDocumentId = false;
                for (int i = 2; i < split.length; i++) {
                    if (split[i].equals("#")) {
                        hasDocumentId = true;
                        break;
                    }
                    String[] indexAndValue = split[i].split(":");
                    // We need to actually create the indices in the data alphabet for all features
                    indices[i - 2] = dataAlphabet.lookupIndex("f" + indexAndValue[0]);
                    features[i - 2] = Double.parseDouble(indexAndValue[1]);
                }
                FeatureVector fv = new FeatureVector(dataAlphabet, indices, features);
                // Now create a label object for the relevance score
                Label label = targetAlphabet.lookupLabel(relevance);
                // Create the actual instance. Its data is the feature vector, its target is the label we created above.
                // The FeatureVector knows the data alphabet, the label knows the target alphabet. The is necessary
                // or the InstanceList won't accept the instance.
                Instance instance = new Instance(fv, label, queryId, hasDocumentId ? split[split.length - 1] : "doc" + documentId);
                ret.add(instance);
                ++documentId;
            }
        }
        return ret;
    }
}

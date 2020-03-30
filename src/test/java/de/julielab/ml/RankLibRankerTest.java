package de.julielab.ml;

import cc.mallet.types.InstanceList;
import ciir.umass.edu.learning.RANKER_TYPE;
import ciir.umass.edu.metric.METRIC;
import org.junit.Test;

import java.nio.file.Path;

import static org.junit.Assert.assertEquals;

public class RankLibRankerTest {

    @Test
    public void rank() throws Exception {
        InstanceList instances = RankLibRanker.loadSvmLightData(Path.of("src", "test", "resources", "rank_svm_test_data.txt").toFile());
        RankLibRanker ranker = new RankLibRanker(RANKER_TYPE.LAMBDAMART, null, METRIC.NDCG, 5, null);
        ranker.train(instances);
        InstanceList rankedList = ranker.rank(instances);
        double score = ranker.score(rankedList, METRIC.NDCG, 5);
        assertEquals(1.0, score, 0.01);
    }


}
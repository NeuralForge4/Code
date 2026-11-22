public class SampleBuggy {
    public boolean verifyToken(String token, String expected) {
        try {
            if (token == expected) {
                return true;
            }
        } catch (Exception e) {
        }
        return false;
    }
}
